import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import pipeline

warnings.filterwarnings("ignore")

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class TextClusterAnalyzer:
    def __init__(
        self,
        model_name="distiluse-base-multilingual-cased",
        llm_model="microsoft/DialoGPT-small",
    ):
        """
        Initialise l'analyseur avec un modèle de sentence transformers et un petit LLM
        """
        print(f"Chargement du modèle d'embeddings {model_name}...")
        self.model = SentenceTransformer(model_name)

        # Initialiser un petit modèle de langage pour la génération de thèmes
        print(f"Chargement du modèle de langage {llm_model}...")
        try:
            # Utiliser un modèle plus petit et efficace pour la génération
            self.theme_generator = pipeline(
                "text-generation",
                model="distilgpt2",  # Modèle léger et rapide
                tokenizer="distilgpt2",
                device=0 if torch.cuda.is_available() else -1,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )
            self.llm_available = True
            print("✅ Modèle de langage chargé avec succès")
        except Exception as e:
            print(f"⚠️ Impossible de charger le modèle de langage: {e}")
            print("🔄 Utilisation du mode de génération basique")
            self.llm_available = False
            self.theme_generator = None

        self.stop_words = set(stopwords.words("french") + stopwords.words("english"))
        self.sentences = []
        self.embeddings = None
        self.labels = None
        self.themes = {}

    def preprocess_text(self, text):
        """
        Prétraite le texte en le divisant en phrases
        """
        # Nettoyer le texte
        text = re.sub(r"\s+", " ", text)  # Normaliser les espaces
        text = text.strip()

        # Diviser en phrases
        sentences = sent_tokenize(text, language="french")

        # Filtrer les phrases trop courtes
        sentences = [s for s in sentences if len(s.split()) > 3]

        self.sentences = sentences
        print(f"Texte divisé en {len(sentences)} phrases.")
        return sentences

    def generate_embeddings(self):
        """
        Génère les embeddings pour chaque phrase
        """
        print("Génération des embeddings...")
        self.embeddings = self.model.encode(self.sentences)
        print(f"Embeddings générés: {self.embeddings.shape}")
        return self.embeddings

    def find_optimal_clusters(self, max_clusters=None, min_clusters=2):
        """
        Trouve automatiquement le nombre optimal de clusters
        """
        n_sentences = len(self.sentences)

        # Validation préliminaire
        if n_sentences < 2:
            print("⚠️ Pas assez de phrases pour optimiser les clusters.")
            return 1

        # Déterminer la plage de clusters à tester
        if max_clusters is None:
            # Règle heuristique : maximum = racine carrée du nombre de phrases
            max_clusters = min(int(np.sqrt(n_sentences)), 10)

        # S'assurer que max_clusters ne dépasse pas le nombre possible
        max_clusters = min(max_clusters, n_sentences - 1)
        min_clusters = max(min_clusters, 2)

        # Validation finale
        if max_clusters < min_clusters:
            print(
                f"⚠️ Pas assez de phrases ({n_sentences}) pour avoir {min_clusters} clusters minimum."
            )
            return max(1, n_sentences - 1)

        if max_clusters == min_clusters:
            print(
                f"📝 Utilisation de {min_clusters} clusters (nombre optimal par défaut)."
            )
            return min_clusters

        print(
            f"🔍 Recherche du nombre optimal de clusters (de {min_clusters} à {max_clusters})..."
        )

        # Méthodes d'évaluation
        silhouette_scores = []
        inertias = []
        cluster_range = range(min_clusters, max_clusters + 1)

        for n_clusters in cluster_range:
            try:
                # Effectuer le clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.embeddings)

                # Calculer le score de silhouette
                silhouette_avg = silhouette_score(self.embeddings, labels)
                silhouette_scores.append(silhouette_avg)

                # Calculer l'inertie (pour la méthode du coude)
                inertias.append(kmeans.inertia_)

                print(f"  📊 {n_clusters} clusters: silhouette = {silhouette_avg:.3f}")

            except Exception as e:
                print(f"  ❌ Erreur avec {n_clusters} clusters: {e}")
                # Ajouter des valeurs par défaut pour maintenir la cohérence
                silhouette_scores.append(0.0)
                inertias.append(float("inf"))

        # Vérifier qu'on a au moins un score valide
        if not silhouette_scores or all(score <= 0 for score in silhouette_scores):
            print(
                "⚠️ Impossible d'optimiser les clusters. Utilisation de la valeur par défaut."
            )
            return min(3, n_sentences - 1)

        # Méthode 1: Meilleur score de silhouette
        best_silhouette_idx = np.argmax(silhouette_scores)
        optimal_silhouette = cluster_range[best_silhouette_idx]

        # Méthode 2: Méthode du coude (elbow method)
        optimal_elbow = self._find_elbow_point(list(cluster_range), inertias)

        # Méthode 3: Analyse de la cohérence des clusters
        optimal_coherence = self._evaluate_cluster_coherence(
            cluster_range, silhouette_scores
        )

        # Choisir le meilleur compromis
        candidates = [optimal_silhouette, optimal_elbow, optimal_coherence]
        candidate_scores = []

        for candidate in candidates:
            if candidate and min_clusters <= candidate <= max_clusters:
                idx = candidate - min_clusters
                if 0 <= idx < len(silhouette_scores):
                    score = silhouette_scores[idx]
                    candidate_scores.append((candidate, score))

        # Sélectionner le candidat avec le meilleur score de silhouette
        if candidate_scores:
            optimal_clusters = max(candidate_scores, key=lambda x: x[1])[0]
        else:
            optimal_clusters = optimal_silhouette

        print(f"✅ Nombre optimal détecté: {optimal_clusters} clusters")
        print(
            f"   🎯 Silhouette: {optimal_silhouette} (score: {silhouette_scores[optimal_silhouette - min_clusters]:.3f})"
        )
        print(f"   📐 Elbow: {optimal_elbow}")
        print(f"   🧠 Cohérence: {optimal_coherence}")

        return optimal_clusters

    def _find_elbow_point(self, cluster_range, inertias):
        """
        Trouve le point de coude dans la courbe d'inertie
        """
        if len(inertias) < 3:
            return None

        # Calculer les différences secondes pour trouver le coude
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)

        # Le coude est où la seconde dérivée est maximale (changement le plus fort)
        if len(second_diffs) > 0:
            elbow_idx = (
                np.argmax(second_diffs) + 2
            )  # +2 car on a perdu 2 points avec les différences
            if elbow_idx < len(cluster_range):
                return cluster_range[elbow_idx]

        return None

    def _evaluate_cluster_coherence(self, cluster_range, silhouette_scores):
        """
        Évalue la cohérence des clusters pour choisir un nombre optimal
        """
        if len(silhouette_scores) < 2:
            return None

        # Chercher un plateau dans les scores de silhouette
        # Un bon nombre de clusters a un score élevé et stable

        best_score = max(silhouette_scores)
        threshold = best_score * 0.95  # 95% du meilleur score

        # Trouver tous les candidats au-dessus du seuil
        good_candidates = []
        for i, score in enumerate(silhouette_scores):
            if score >= threshold:
                good_candidates.append(cluster_range[i])

        if good_candidates:
            # Préférer un nombre modéré de clusters (ni trop peu, ni trop)
            mid_point = len(cluster_range) / 2
            distances = [abs(c - mid_point) for c in good_candidates]
            best_idx = np.argmin(distances)
            return good_candidates[best_idx]

        return None

    def cluster_sentences(self, n_clusters=None, auto_detect=True):
        """
        Regroupe les phrases en clusters avec détection automatique du nombre optimal
        """
        n_sentences = len(self.sentences)

        # Validation du nombre de phrases
        if n_sentences < 2:
            print(
                "⚠️ Pas assez de phrases pour effectuer un clustering (minimum 2 requises)."
            )
            # Créer un cluster unique
            self.labels = np.zeros(n_sentences, dtype=int)
            return self.labels

        if auto_detect and n_clusters is None:
            # Détection automatique du nombre optimal de clusters
            n_clusters = self.find_optimal_clusters()
        elif n_clusters is None:
            # Valeur par défaut si pas de détection automatique
            n_clusters = min(5, n_sentences - 1)  # Ne peut pas dépasser n_sentences-1

        # Validation critique : s'assurer que n_clusters <= n_sentences
        if n_clusters >= n_sentences:
            print(
                f"⚠️ Ajustement: {n_clusters} clusters demandés mais seulement {n_sentences} phrases disponibles."
            )
            n_clusters = max(1, n_sentences - 1)  # Maximum possible
            if n_clusters == 1:
                print(
                    "📝 Création d'un cluster unique (pas assez de phrases pour diviser)."
                )
                self.labels = np.zeros(n_sentences, dtype=int)
                return self.labels

        print(f"🎯 Clustering en {n_clusters} groupes...")

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = kmeans.fit_predict(self.embeddings)

            # Calculer et afficher les métriques de qualité
            if len(set(self.labels)) > 1:  # Vérifier qu'il y a plusieurs clusters
                silhouette_avg = silhouette_score(self.embeddings, self.labels)
                print(f"📊 Score de silhouette: {silhouette_avg:.3f}")

        except Exception as e:
            print(f"❌ Erreur lors du clustering: {e}")
            print("🔄 Création d'un cluster unique en fallback.")
            self.labels = np.zeros(n_sentences, dtype=int)
            return self.labels

        # Afficher la distribution des clusters
        cluster_counts = Counter(self.labels)
        print("📈 Distribution des clusters:")
        for cluster_id, count in sorted(cluster_counts.items()):
            percentage = (count / len(self.sentences)) * 100
            print(f"  Cluster {cluster_id}: {count} phrases ({percentage:.1f}%)")

        return self.labels

    def extract_cluster_themes(self):
        """
        Analyse chaque cluster pour proposer un thème
        """
        print("Extraction des thèmes par cluster...")

        for cluster_id in np.unique(self.labels):
            # Récupérer les phrases du cluster
            cluster_sentences = [
                self.sentences[i]
                for i in range(len(self.sentences))
                if self.labels[i] == cluster_id
            ]

            # Combiner toutes les phrases du cluster
            cluster_text = " ".join(cluster_sentences)

            # Extraire les mots-clés avec TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=1,
            )

            try:
                # Comparer avec tous les autres textes pour avoir un bon TF-IDF
                all_texts = []
                for cid in np.unique(self.labels):
                    other_sentences = [
                        self.sentences[i]
                        for i in range(len(self.sentences))
                        if self.labels[i] == cid
                    ]
                    all_texts.append(" ".join(other_sentences))

                tfidf_matrix = vectorizer.fit_transform(all_texts)
                feature_names = vectorizer.get_feature_names_out()

                # Récupérer les scores TF-IDF pour ce cluster
                cluster_idx = list(np.unique(self.labels)).index(cluster_id)
                tfidf_scores = tfidf_matrix[cluster_idx].toarray()[0]

                # Obtenir les top mots-clés
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                keywords = [
                    feature_names[i] for i in top_indices if tfidf_scores[i] > 0
                ]

                # Générer un thème basé sur les mots-clés et l'analyse sémantique
                if keywords:
                    theme = self.generate_theme_from_keywords(
                        keywords, cluster_sentences
                    )
                else:
                    # Même sans mots-clés TF-IDF, essayer d'extraire un thème des phrases
                    theme = self._extract_theme_from_sentences(cluster_sentences)

                self.themes[cluster_id] = {
                    "theme": theme,
                    "keywords": keywords[:5],
                    "sentences_count": len(cluster_sentences),
                    "sample_sentence": cluster_sentences[0]
                    if cluster_sentences
                    else "",
                }

            except Exception as e:
                print(f"Erreur lors de l'analyse du cluster {cluster_id}: {e}")
                # Même en cas d'erreur, essayer d'extraire un thème des phrases
                cluster_sentences = [
                    self.sentences[i]
                    for i in range(len(self.sentences))
                    if self.labels[i] == cluster_id
                ]
                theme = self._extract_theme_from_sentences(cluster_sentences)

                self.themes[cluster_id] = {
                    "theme": theme,
                    "keywords": [],
                    "sentences_count": len(cluster_sentences),
                    "sample_sentence": cluster_sentences[0]
                    if cluster_sentences
                    else "",
                }

        return self.themes

    def generate_theme_from_keywords(self, keywords, sentences):
        """
        Génère automatiquement un nom de thème en utilisant un modèle de langage
        """
        if not keywords:
            return "Thème Général"

        if self.llm_available:
            return self._generate_theme_with_llm(keywords, sentences)
        else:
            return self._generate_theme_fallback(keywords, sentences)

    def _generate_theme_with_llm(self, keywords, sentences):
        """
        Utilise un petit modèle de langage pour générer un thème naturel
        """
        try:
            # Préparer le contexte pour le modèle
            context_keywords = ", ".join(keywords[:5])
            sample_sentence = sentences[0][:100] if sentences else ""

            # Créer plusieurs prompts pour avoir de la variété
            prompts = [
                f"Based on keywords: {context_keywords}. Theme:",
                f"Topic about: {context_keywords}. Main theme:",
                f"Subject: {context_keywords}. Category:",
            ]

            best_theme = None
            best_score = 0

            for prompt in prompts:
                try:
                    # Générer avec le modèle
                    generated = self.theme_generator(
                        prompt,
                        max_length=len(prompt.split()) + 8,  # Limiter la génération
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=50256,
                    )

                    # Extraire le thème généré
                    full_text = generated[0]["generated_text"]
                    theme_part = full_text.replace(prompt, "").strip()

                    # Nettoyer le thème généré
                    theme = self._clean_generated_theme(theme_part, keywords)

                    # Évaluer la qualité du thème généré
                    score = self._evaluate_theme_quality(theme, keywords)

                    if score > best_score:
                        best_score = score
                        best_theme = theme

                except Exception:
                    continue

            # Si on a un bon thème généré, l'utiliser
            if best_theme and best_score > 0.3:
                return best_theme

        except Exception as e:
            print(f"Erreur lors de la génération avec LLM: {e}")

        # Fallback vers la méthode alternative
        return self._generate_theme_fallback(keywords, sentences)

    def _clean_generated_theme(self, raw_theme, keywords):
        """
        Nettoie et améliore le thème généré par le LLM
        """
        if not raw_theme:
            return None

        # Supprimer les caractères indésirables
        theme = re.sub(r"[^\w\s&-]", "", raw_theme)
        theme = re.sub(r"\s+", " ", theme).strip()

        # Prendre seulement les premiers mots (éviter les phrases trop longues)
        words = theme.split()[:4]  # Maximum 4 mots
        theme = " ".join(words)

        # Capitaliser proprement
        if theme:
            # Capitaliser chaque mot important
            theme = " ".join(
                word.capitalize() if len(word) > 2 else word for word in theme.split()
            )

        return theme if theme and len(theme) > 2 else None

    def _evaluate_theme_quality(self, theme, keywords):
        """
        Évalue la qualité d'un thème généré
        """
        if not theme or len(theme) < 3:
            return 0

        score = 0
        theme_lower = theme.lower()

        # Points pour la pertinence avec les mots-clés
        for keyword in keywords[:3]:
            if any(part.lower() in theme_lower for part in keyword.split()):
                score += 0.3

        # Points pour la longueur appropriée
        word_count = len(theme.split())
        if 1 <= word_count <= 4:
            score += 0.3

        # Points pour l'absence de mots parasites
        bad_words = [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
        ]
        if not any(bad in theme_lower for bad in bad_words):
            score += 0.2

        # Points pour la cohérence (pas de répétitions)
        words = theme.split()
        if len(set(word.lower() for word in words)) == len(words):
            score += 0.2

        return min(score, 1.0)

    def _generate_theme_fallback(self, keywords, sentences):
        """
        Méthode de fallback pour générer des thèmes sans LLM
        """
        # Méthode basée sur l'analyse sémantique intelligente
        primary_keywords = keywords[:3]

        # Analyser les patterns sémantiques
        theme_candidates = self._extract_semantic_themes(primary_keywords, sentences)

        # Générer un thème intelligent
        return self._create_intelligent_theme(primary_keywords, theme_candidates)

    def _create_intelligent_theme(self, keywords, context_words):
        """
        Crée un thème intelligent basé sur l'analyse sémantique
        """
        all_words = keywords + context_words

        # Essayer de créer un thème conceptuel
        conceptual_theme = self._create_conceptual_theme(all_words)
        if conceptual_theme:
            return conceptual_theme

        # Sinon, combiner intelligemment les mots-clés
        return self._combine_keywords_creatively(keywords)

    def _create_conceptual_theme(self, words):
        """
        Crée un thème conceptuel basé sur l'analyse des mots
        """
        word_string = " ".join(words).lower()

        # Identifier des concepts abstraits
        concept_indicators = {
            "innovation": [
                "nouveau",
                "innovation",
                "technologie",
                "développement",
                "avancé",
            ],
            "transformation": [
                "changement",
                "évolution",
                "transformation",
                "mutation",
                "adaptation",
            ],
            "analyse": ["étude", "recherche", "analyse", "examen", "investigation"],
            "gestion": [
                "management",
                "gestion",
                "organisation",
                "administration",
                "contrôle",
            ],
            "développement": [
                "croissance",
                "expansion",
                "progrès",
                "amélioration",
                "développement",
            ],
            "communication": [
                "information",
                "communication",
                "message",
                "dialogue",
                "échange",
            ],
            "stratégie": [
                "planification",
                "stratégie",
                "tactique",
                "approche",
                "méthode",
            ],
        }

        # Calculer les scores conceptuels
        concept_scores = {}
        for concept, indicators in concept_indicators.items():
            score = sum(1 for indicator in indicators if indicator in word_string)
            if score > 0:
                concept_scores[concept] = score

        # Combiner avec les mots principaux
        if concept_scores:
            best_concept = max(concept_scores.items(), key=lambda x: x[1])[0]
            main_word = words[0] if words else best_concept
            return f"{main_word.title()} & {best_concept.title()}"

        return None

    def _combine_keywords_creatively(self, keywords):
        """
        Combine créativement les mots-clés pour un thème original
        """
        if not keywords:
            return "Thème Émergent"

        # Nettoyer les mots-clés
        clean_words = []
        for kw in keywords[:2]:
            words = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", kw)
            clean_words.extend([w for w in words if len(w) > 2])

        if not clean_words:
            return "Concept Principal"

        # Créer des combinaisons créatives
        if len(clean_words) == 1:
            return f"Domaine {clean_words[0].title()}"
        elif len(clean_words) >= 2:
            # Essayer différentes combinaisons
            combinations = [
                f"{clean_words[0].title()} & {clean_words[1].title()}",
                f"Enjeux {clean_words[0].title()}",
                f"Univers {clean_words[0].title()}",
                f"{clean_words[0].title()} Moderne",
            ]

            # Choisir la combinaison la plus équilibrée
            return combinations[0]  # Préférer la combinaison simple

        return "Thématique Spécialisée"

    def _extract_semantic_themes(self, keywords, sentences):
        """
        Extrait des thèmes potentiels en analysant le contexte sémantique
        """
        # Analyser les mots qui apparaissent fréquemment avec les mots-clés
        context_words = []

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [
                w
                for w in words
                if w.isalpha() and w not in self.stop_words and len(w) > 2
            ]

            # Chercher les mots qui apparaissent près des mots-clés
            for keyword in keywords:
                if any(kw_part in sentence.lower() for kw_part in keyword.split()):
                    context_words.extend(words)

        # Compter les cooccurrences
        context_counter = Counter(context_words)

        # Extraire les mots de contexte les plus fréquents (excluant les mots-clés originaux)
        filtered_context = []
        for word, count in context_counter.most_common(10):
            if not any(word in kw.lower() or kw.lower() in word for kw in keywords):
                filtered_context.append(word)

        return filtered_context[:5]

    def _generate_automatic_theme(self, primary_keywords, context_words):
        """
        Génère automatiquement un thème cohérent
        """
        all_theme_words = primary_keywords + context_words

        # Stratégie 1: Identifier des concepts centraux
        theme_name = self._identify_central_concept(all_theme_words)

        if theme_name:
            return theme_name

        # Stratégie 2: Combiner les mots-clés principaux de manière intelligente
        return self._combine_keywords_intelligently(primary_keywords)

    def _extract_theme_from_sentences(self, sentences):
        """
        Extrait un thème directement des phrases quand les mots-clés TF-IDF ne suffisent pas
        """
        if not sentences:
            return "Thème Général"

        # Extraire tous les mots significatifs des phrases
        all_words = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [
                w
                for w in words
                if w.isalpha() and w not in self.stop_words and len(w) > 3
            ]
            all_words.extend(words)

        # Compter les fréquences
        word_freq = Counter(all_words)
        top_words = [word for word, freq in word_freq.most_common(5)]

        # Utiliser la même logique de génération automatique
        if top_words:
            return self._create_intelligent_theme(top_words, [])

        return "Thème Général"

    def _identify_central_concept(self, words):
        """
        Identifie un concept central à partir des mots disponibles
        """
        # Analyser les patterns sémantiques automatiquement
        word_string = " ".join(words).lower()

        # Détection de domaines par analyse de patterns (extensible)
        domain_patterns = {
            # Patterns économiques
            "économie": [
                "marché",
                "finance",
                "entreprise",
                "investissement",
                "commercial",
                "business",
                "économique",
                "financier",
                "prix",
                "coût",
                "profit",
                "industrie",
            ],
            # Patterns technologiques
            "technologie": [
                "digital",
                "numérique",
                "algorithme",
                "données",
                "intelligence",
                "artificielle",
                "technologique",
                "innovation",
                "système",
                "logiciel",
                "ordinateur",
            ],
            # Patterns de santé
            "santé": [
                "médical",
                "patient",
                "traitement",
                "maladie",
                "diagnostic",
                "thérapie",
                "clinique",
                "hôpital",
                "médecin",
                "soins",
                "médicament",
            ],
            # Patterns éducatifs
            "éducation": [
                "apprentissage",
                "étudiant",
                "formation",
                "enseignement",
                "pédagogique",
                "éducatif",
                "cours",
                "école",
                "université",
                "connaissance",
            ],
            # Patterns environnementaux
            "environnement": [
                "climatique",
                "écologique",
                "nature",
                "environnemental",
                "durable",
                "climat",
                "écologie",
                "planète",
                "biodiversité",
                "énergie",
            ],
            # Patterns sociaux
            "société": [
                "social",
                "communauté",
                "population",
                "citoyen",
                "public",
                "collectif",
                "humain",
                "culturel",
                "sociétal",
                "groupe",
            ],
            # Patterns politiques
            "politique": [
                "gouvernement",
                "politique",
                "état",
                "public",
                "administration",
                "pouvoir",
                "autorité",
                "institution",
                "régulation",
            ],
        }

        # Calculer les scores de correspondance
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in word_string)
            if score > 0:
                domain_scores[domain] = score

        # Retourner le domaine avec le meilleur score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            return best_domain.title()

        return None

    def _combine_keywords_intelligently(self, keywords):
        """
        Combine intelligemment les mots-clés pour créer un thème cohérent
        """
        if not keywords:
            return "Thème Général"

        # Nettoyer et préparer les mots-clés
        clean_keywords = []
        for kw in keywords[:2]:  # Prendre les 2 premiers mots-clés
            # Enlever les caractères spéciaux et diviser les mots composés
            clean_words = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", kw)
            clean_keywords.extend(clean_words)

        # Supprimer les doublons tout en préservant l'ordre
        seen = set()
        unique_keywords = []
        for kw in clean_keywords:
            if kw.lower() not in seen and len(kw) > 2:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        if not unique_keywords:
            return "Thème Général"

        # Stratégies de combinaison
        if len(unique_keywords) == 1:
            return unique_keywords[0].title()
        elif len(unique_keywords) == 2:
            return f"{unique_keywords[0].title()} & {unique_keywords[1].title()}"
        else:
            # Pour 3+ mots, prendre les 2 plus significatifs
            main_concept = unique_keywords[0].title()
            secondary = unique_keywords[1].title()
            return f"{main_concept} & {secondary}"

    def visualize_clusters(self, figsize=(12, 8)):
        """
        Visualise les clusters avec t-SNE
        """
        print("Génération de la visualisation t-SNE...")

        # Réduire la dimensionnalité avec t-SNE
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(self.sentences) - 1)
        )
        embeddings_2d = tsne.fit_transform(self.embeddings)

        # Créer le graphique
        plt.figure(figsize=figsize)

        # Couleurs pour chaque cluster
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(self.labels))))

        # Tracer chaque cluster
        for cluster_id in np.unique(self.labels):
            mask = self.labels == cluster_id
            theme_name = self.themes[cluster_id]["theme"]

            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[cluster_id]],
                label=f"Cluster {cluster_id}: {theme_name}",
                alpha=0.7,
                s=50,
            )

            # Ajouter le label du thème au centre du cluster
            cluster_center = embeddings_2d[mask].mean(axis=0)
            plt.annotate(
                theme_name,
                cluster_center,
                xytext=(5, 5),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor=colors[cluster_id], alpha=0.7
                ),
                fontsize=9,
                ha="center",
            )

        plt.title("Visualisation t-SNE des Clusters de Phrases", fontsize=16, pad=20)
        plt.xlabel("Dimension t-SNE 1")
        plt.ylabel("Dimension t-SNE 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_analysis_summary(self):
        """
        Affiche un résumé de l'analyse
        """
        print("\n" + "=" * 50)
        print("RÉSUMÉ DE L'ANALYSE")
        print("=" * 50)

        for cluster_id, info in self.themes.items():
            print(f"\n🎯 CLUSTER {cluster_id}: {info['theme']}")
            print(f"   📊 Nombre de phrases: {info['sentences_count']}")
            print(f"   🔑 Mots-clés: {', '.join(info['keywords'])}")
            print(f"   📝 Exemple: {info['sample_sentence'][:100]}...")

    def analyze(self, text, n_clusters=None, auto_detect=True):
        """
        Lance l'analyse complète avec détection automatique du nombre de clusters
        """
        print("🚀 Début de l'analyse...")

        # Étapes de l'analyse
        self.preprocess_text(text)
        self.generate_embeddings()

        # Clustering avec détection automatique
        if auto_detect:
            print("🔍 Mode détection automatique activé")

        self.cluster_sentences(n_clusters, auto_detect)
        self.extract_cluster_themes()

        # Affichage des résultats
        self.print_analysis_summary()
        self.visualize_clusters()

        return self.themes


# Exemple d'utilisation
if __name__ == "__main__":
    # Texte d'exemple
    sample_text = """
    L'intelligence artificielle transforme notre société de manière profonde. Les algorithmes d'apprentissage automatique 
    permettent aux machines de comprendre et d'analyser des données complexes. Cette révolution technologique impacte 
    de nombreux secteurs d'activité.
    
    En économie, les marchés financiers utilisent des algorithmes de trading haute fréquence. Les entreprises investissent 
    massivement dans la transformation numérique. Les cryptomonnaies bouleversent les systèmes de paiement traditionnels. 
    L'analyse prédictive aide les entreprises à optimiser leurs stratégies.
    
    Dans le domaine de la santé, l'IA permet de diagnostiquer des maladies plus rapidement. Les médecins utilisent des 
    outils d'aide au diagnostic basés sur l'apprentissage automatique. La télémédecine se développe grâce aux nouvelles 
    technologies. Les patients bénéficient de traitements personnalisés.
    
    L'éducation connaît également une transformation majeure. Les plateformes d'apprentissage en ligne se multiplient. 
    Les étudiants accèdent à des cours personnalisés grâce à l'IA. Les enseignants utilisent des outils numériques 
    pour améliorer leurs méthodes pédagogiques.
    
    Cependant, ces changements soulèvent des questions éthiques importantes. La protection des données personnelles 
    devient cruciale. Les biais algorithmiques peuvent créer des discriminations. Il faut réguler l'usage de ces 
    technologies pour protéger les citoyens.
    
    Le changement climatique représente un défi majeur pour l'humanité. Les émissions de gaz à effet de serre continuent 
    d'augmenter malgré les accords internationaux. Les énergies renouvelables se développent mais pas assez rapidement. 
    Les gouvernements doivent accélérer la transition écologique.
    """

    sample_text = """Bibliothèque - Curis -au -Mont -d'Or

 Accueil
 Le village

 Présentation
 Histoire
 Patrimoine
 Photothèque

 Curis -au -Mont -d’Or hier
 Curis -au -Mont -d’Or aujourd’hui
 Curis -au -Mont -d’Or - Photos des évènements

 Vie économique
 Projets

 Projet réalisés Terrain des poiriers
 APPEL A PROJETS - BAR/RESTAURANT
 Projet réalisé Engazonnement Cimetière
 Projet réalisé micro -crèche

 Plans

 Mairie

 Les Elus
 Le personnel communal
 Finances
 Urbanisme
 Equipements municipaux
 Commissions/syndicats
 Comptes -rendus des conseils municipaux
 Les arrêtés municipaux temporaires
 Les arrêtés municipaux permanents
 Le Têtu
 Le Tambour

 Enfance et Social

 L’école publique

 L’école
 Le restaurant scolaire
 Le périscolaire
 Portail Famille : inscriptions

 Relais d’assistantes maternelles
 Les centres de loisirs
 CCAS (Centre communal d’action sociale)
 AIAD -Aide à domicile
 Service de repas à domicile
 Micro crèche

 Animation et Culture

 Bibliothèque
 Associations

 Association communale de chasse
 Comité des fêtes de Curis
 Iaido
 Body Karaté -Karaté Défense
 Sou des écoles
 Sports et Loisirs
 ThouAMAPorte
 A Thouboutdechamps (jardin partagé)
 De Thou Choeur

 Balades
 Planning des activités
 Les événements récurrents à Curis

 Cadre de Vie

 Environnement
 Métropole
 Collecte et traitement des déchets - gros appareils ménagers
 Comment obtenir une poubelle ?
 Vivre ensemble

 Propreté canine
 Plantations (haies, arbres, arbustes…)
 Nuisances sonores / bruits de voisinage
 Déneigement
 Brûlage des déchets verts
 Chiens en divagation
 Chenille processionnaire du pin

 Syndicat Mixte Plaines Monts d’Or
 Plan Climat Communal
 Ambroisie

 Infos pratiques

 Formalités administratives
 Le recensement militaire
 Numéros d’urgence
 Demandes et réclamations communautaires
 Infos transport
 Santé à Curis
 Mutuelle Santé
 Réserver la salle du Vallon
 Le cimetière
 Prêt de mobilier extérieur
 Correspondant Progrès à Curis
 Annoncer un événement

 Contact

 Accueil " Animation et Culture " Bibliothèque

 Acquisitions 2017 Beaton M La quiche fatale
 Bonnefoy M Sucre noir
 Bourdin F L'homme de leur vie
 Cayre H La daronne
 Chalandon Sorj Le jour d'avant
 Chantraine O Un élément perturbateur
 Chavassieux C La vie volée de Martin
 Cognetti P Les huit montagnes
 De Giovanni M Le noel du commissaire Ricciardi
 Deghelt F Agatha
 Deliry J La maraude
 Deserable F -H Un certain Monsieur Piekielny
 Djian P Marlene
 Ducrozet P L'invention des corps
 Dupuis M -B Les amants du presbytere
 Ellory R J Un cœur sombre
 Ferrante E L'amie prodigieuse (3tomes)
 Gavalda A Fendre l'armure
 Giesbert F O Belle d'amour
 Giraud B Un loup pour l'homme
 Guez O La disparition de Josef Mengele
 Haenel Y Tiens ferme ta couronne
 Jaenada P La serpe
 Kemeid O Tangvald
 Khadra Y Ce que le mirage doit à l'oasis
 Larsson A En sacrifice à Moloch
 Ledig A De tes nouvelles
 Lenglet A Temps de haine
 Leon D Minuit sur le canal San Boldo
 Malte M Le garçon
 Montero Manglano L La table du roi Salomon
 Nguyen V T Le sympathisant
 Nohant G Légende d'un dormeur éveillé
 Olafsdottir A A Or
 Pamuk O Cette chose étrange en moi
 Pennac D Le cas Malaussene
 Recondo L Point cardinal
 Rufin J C Le tour du monde du roi Zibeline
 Sabolo M Summer
 Schwarz -Bart S Adieu Bogota
 Signol C La vie en son royaume
 Suter M Elephant
 Van Cauwelaert D Le retour de Jules
 Vargas F Quand sort la recluse
 Viel T Article 353 du code penal
 Wideman J -E Ecrire pour sauver une vie
 Yun M Les ames des enfants endormis
 Zeniter A L'art de perdre

 Bibliothèque

 ACTUALITE
 La Bibliothèque Municipale de Lyon (BML) est notre nouveau partenaire pour la fourniture d’un fond de 350 ouvrages disponible depuis le 15 mars 2018 .
 Retrouvez -les en téléchargeant la liste ici: Bibliothèque municipale de Lyon ou venez les emprunter les vendredis (16h00 -17h30 ) et samedis matins (10h -12h).
 Vous pouvez aussi consulter:
 la liste des ouvrages acquis pour le fonds propre de la bibliothèque municipale de Curis

 Acquisitions 2016
 Acquisitions 2017
 Acquisitions 2018
 Acquisitions 2020
 Acquisitions

 Les listes, par genres des ouvrages disponibles à la Bibliothèque de Curis:
 Bandes dessinées Documentaires Fantasia Livres en gros caractères Romans
 romans policiers
 La bibliothèque, comment ça marche?
 Présentation
 Objectif :

 Mettre à la disposition du public adulte des romans, des documents, des biographies.

 Moyens :

 Un budget annuel de 1200 euros pour acquérir des livres.
 Un dépôt de la Bibliothèque Municipale de Lyon renouvelé régulièrement.
 Un fonds propre d’un millier d’ouvrages.

 Enjeux :

 Proposer régulièrement des romans contemporains et échanger des avis sur les lectures.

 Acteurs :

 Un groupe de bénévoles passionnés par la lecture.

 Infos pratiques
 Quand ?
Les permanences ont lieu tous les samedis matin de 10 heures à midi et les vendredis (hors vacances scolaires) de 16h00 à 17h30
 Où ?
Dans l’aile gauche de la Mairie, côté garderie, rue de la Mairie.
 Comment s’inscrire ?
Si vous disposez d’une adresse mail valide, il vous suffit lors de votre première visite, de remplir une fiche d’inscription. Si vous ne disposez pas d’une adresse mail veuillez apporter 5 enveloppes timbrées et à votre adresse.
 Emprunts : mode d’emploi
 Vous pouvez emprunter autant de livres que vous le souhaitez (mais avec un maximum de deux nouveautés) et pour une durée de trois semaines.
Nous pouvons vous “commander” des titres précis, des auteurs, des thèmes,… auprès de la BML.
 Notre partenaire, la BML (Bibliothèque Municipale de Lyon) :
Vous pouvez consulter le catalogue sur le site de la BML et nous demander de réserver les ouvrages qui vous intéressent. Une navette nous livre tous les mois les ouvrages disponibles.
 Comité de lecture :
 Un groupe de lecture ouvert à tous et associant les bibliothèques d’Albigny, Poleymieux, Saint Germain, Couzon et Curis se réunit tous les deux mois et permet d’échanger des avis sur un choix de livres.

 Mairie

 Adresse :
Rue de la Mairie
69250 Curis -au -Mont -d'Or

 Contact :
Téléphone : 04 .78 .91 .24 .02
Télécopie : 04 .78 .98 .28 .05
Courriel : mairie@curis.fr

 Facebook
 Mentions légales
 Plan du site
 Crédits
 Contact

 Nous utilisons des cookies pour vous garantir la meilleure expérience sur notre site. Si vous continuez à utiliser ce dernier, nous considérerons que vous acceptez l'utilisation des cookies. Ok

 Accueil
 Le village

 Présentation
 Histoire
 Patrimoine
 Photothèque

 Curis -au -Mont -d’Or hier
 Curis -au -Mont -d’Or aujourd’hui
 Curis -au -Mont -d’Or - Photos des évènements

 Vie économique
 Projets

 Projet réalisés Terrain des poiriers
 APPEL A PROJETS - BAR/RESTAURANT
 Projet réalisé Engazonnement Cimetière
 Projet réalisé micro -crèche

 Plans

 Mairie

 Les Elus
 Le personnel communal
 Finances
 Urbanisme
 Equipements municipaux
 Commissions/syndicats
 Comptes -rendus des conseils municipaux
 Les arrêtés municipaux temporaires
 Les arrêtés municipaux permanents
 Le Têtu
 Le Tambour

 Enfance et Social

 L’école publique

 L’école
 Le restaurant scolaire
 Le périscolaire
 Portail Famille : inscriptions

 Relais d’assistantes maternelles
 Les centres de loisirs
 CCAS (Centre communal d’action sociale)
 AIAD -Aide à domicile
 Service de repas à domicile
 Micro crèche

 Animation et Culture

 Bibliothèque
 Associations

 Association communale de chasse
 Comité des fêtes de Curis
 Iaido
 Body Karaté -Karaté Défense
 Sou des écoles
 Sports et Loisirs
 ThouAMAPorte
 A Thouboutdechamps (jardin partagé)
 De Thou Choeur

 Balades
 Planning des activités
 Les événements récurrents à Curis

 Cadre de Vie

 Environnement
 Métropole
 Collecte et traitement des déchets - gros appareils ménagers
 Comment obtenir une poubelle ?
 Vivre ensemble

 Propreté canine
 Plantations (haies, arbres, arbustes…)
 Nuisances sonores / bruits de voisinage
 Déneigement
 Brûlage des déchets verts
 Chiens en divagation
 Chenille processionnaire du pin

 Syndicat Mixte Plaines Monts d’Or
 Plan Climat Communal
 Ambroisie

 Infos pratiques

 Formalités administratives
 Le recensement militaire
 Numéros d’urgence
 Demandes et réclamations communautaires
 Infos transport
 Santé à Curis
 Mutuelle Santé
 Réserver la salle du Vallon
 Le cimetière
 Prêt de mobilier extérieur
 Correspondant Progrès à Curis
 Annoncer un événement

 Contact"""

    # Créer et lancer l'analyseur avec génération LLM et détection automatique
    print("🤖 Initialisation avec modèle de langage pour génération de thèmes...")
    analyzer = TextClusterAnalyzer()

    # Analyse avec détection automatique du nombre de clusters
    print("\n🔍 MÉTHODE 1: Détection automatique du nombre de clusters")
    themes_auto = analyzer.analyze(sample_text, auto_detect=True)

    # Exemple avec nombre fixe pour comparaison
    print("\n\n🎯 MÉTHODE 2: Nombre de clusters fixé à 4 (pour comparaison)")
    analyzer2 = TextClusterAnalyzer()
    themes_fixed = analyzer2.analyze(sample_text, n_clusters=4, auto_detect=False)

    print("\n" + "=" * 60)
    print("💡 INFORMATIONS TECHNIQUES")
    print("=" * 60)
    print("🔹 Embeddings: sentence-transformers (analyse sémantique)")
    print("🔹 Clustering: K-means avec détection automatique optimale")
    print("🔹 Optimisation: Score de silhouette + Méthode du coude + Cohérence")
    print("🔹 Génération de thèmes: DistilGPT-2 (LLM léger)")
    print("🔹 Visualisation: t-SNE (réduction dimensionnelle)")
    print("🔹 Fallback: Analyse sémantique intelligente")

    # Instructions d'installation
    print("\n📦 DÉPENDANCES REQUISES:")
    print(
        "pip install sentence-transformers scikit-learn matplotlib nltk transformers torch"
    )

    print("\n🎛️ UTILISATION:")
    print("# Détection automatique (recommandé)")
    print("analyzer.analyze(text)")
    print("# Nombre fixe")
    print("analyzer.analyze(text, n_clusters=5, auto_detect=False)")
