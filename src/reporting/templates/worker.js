// ForzaEmbed — Web Worker: decompress and parse report data
// Loads pako (zlib) from CDN for Base64+zlib decompression

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js');

self.onmessage = function (event) {
    const base64String = event.data;
    try {
        // Step 1: Decode Base64 to binary string
        const binaryString = atob(base64String);

        // Step 2: Convert binary string to Uint8Array
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Step 3: Decompress using pako (zlib inflate)
        const decompressedBytes = pako.inflate(bytes);

        // Step 4: Decode UTF-8 bytes to string
        const jsonString = new TextDecoder('utf-8').decode(decompressedBytes);

        // Step 5: Parse JSON and send result back
        const processedData = JSON.parse(jsonString);
        self.postMessage({ success: true, data: processedData });
    } catch (e) {
        self.postMessage({ success: false, error: e.message });
    }
};
