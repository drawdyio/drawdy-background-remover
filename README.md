# Background Remover

Select one or more images, right-click, and choose
**Remove background**. Segmentation runs inside the driver worker on ONNX
Runtime Web's WebAssembly backend; no pixels leave the browser.

![example](https://github.com/drawdyio/drawdy-background-remover/blob/main/selecting.png?raw=true)

![example2](https://github.com/drawdyio/drawdy-background-remover/blob/main/done.png?raw=true)


## Model

`u2netp` (4.6 MB), U²-Net (Apache-2.0), rembg export. It is fetched from a
CORS-enabled Hugging Face mirror of rembg's release assets and verified against
a pinned SHA-256 before a session is created. The worker has an opaque origin,
so the download cannot be cached across page loads; the model is fetched once
per session and reused for every image.



