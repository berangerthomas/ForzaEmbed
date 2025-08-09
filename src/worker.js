// Import the pako library for zlib decompression
self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js');

self.onmessage = function(event) {
    const base64String = event.data;
    try {
        // Directly decompress the base64 decoded string.
        // atob() decodes base64 to a binary string, which pako can handle.
        const jsonString = pako.inflate(atob(base64String), { to: 'string' });
        const processedData = JSON.parse(jsonString);
        
        // Send the processed data back to the main thread
        self.postMessage({ success: true, data: processedData });
    } catch (e) {
        // Report an error back to the main thread
        self.postMessage({ success: false, error: e.message });
    }
};
