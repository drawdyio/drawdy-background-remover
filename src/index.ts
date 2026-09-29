import type {
  DomElementSchema,
  DriverCommandIssuer,
  DriverModule,
  ModuleStyling,
  ProtocolCommandError,
} from "@drawdy/driver-protocol";
import type { InferenceSession } from "onnxruntime-web";
import {
  applyAlpha,
  bytesToHex,
  fitWithin,
  maskFromOutput,
  MODEL_INPUT_SIZE,
  preprocess,
  resizeMaskBilinear,
  type Size,
} from "./segment";

const ORT_VERSION = "1.29.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_SCRIPT = `${ORT_BASE}ort.wasm.min.js`;
const MODEL_INPUT_NAME = "input.1";
const MAX_SIDE = 4096;

const RESULT_GAP = 40;
const MENU_ID = "background-remover";
const TOAST_ID = "background-remover:toast";
const TOAST_WIDTH = 360;

const MODEL = {
  url: "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2netp.onnx",
  sha256: "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
  sizeMb: 5,
};

let issue: DriverCommandIssuer;
let driverId: string;
let styling: ModuleStyling;
let seq = 0;
let generateId: () => string = () => "";
const rid = (): string => String(seq++);

let sessionPromise: Promise<InferenceSession> | null = null;
let ortLoaded = false;
let toastToken = 0;

let queue = Promise.resolve();
const enqueue = (f: () => Promise<void>): void => {
  queue = queue.then(f).catch(() => undefined);
};

export const activate: DriverModule["activate"] = async (ctx) => {
  issue = ctx.issueCommand;
  driverId = ctx.manifest.driverId;
  styling = ctx.styling;
  generateId = ctx.generateId;

  await issue({
    type: "command:context-menu:add",
    driverId,
    requestId: rid(),
    req: {
      menuId: MENU_ID,
      menuTitle: "Remove background",
    },
  });
  await issue({
    type: "subscription:context-menu:clicked",
    driverId,
    requestId: rid(),
    req: { menuId: MENU_ID },
  });
  await issue({
    type: "subscription:dom:theme-changed",
    driverId,
    requestId: rid(),
  });
};

export const onEvent: DriverModule["onEvent"] = async (event) => {
  if (event.type === "subscription:dom:theme-changed") {
    styling = event.body.styling;
    return;
  }
  if (event.type === "subscription:context-menu:clicked") {
    if (event.body.menuId === MENU_ID) enqueue(run);
  }
};

async function run(): Promise<void> {
  const imageIds = await selectedImageIds();
  if (imageIds.length === 0) {
    await toast("Select an image first", 2500);
    return;
  }
  try {
    const session = await ensureSession();
    for (let i = 0; i < imageIds.length; i++) {
      const progress =
        imageIds.length > 1 ? ` (${i + 1}/${imageIds.length})` : "";
      await toast(`Removing background${progress}…`);
      await processImage(session, imageIds[i]);
    }
    const noun = imageIds.length === 1 ? "image" : "images";
    await toast(`Background removed from ${imageIds.length} ${noun}`, 2000);
  } catch (err) {
    await toast(`Background removal failed: ${describe(err)}`, 5000);
  }
}

async function selectedImageIds(): Promise<string[]> {
  const selected = unwrap(
    await issue({
      type: "command:scene:get-current-selected-drawdy-elements",
      driverId,
      requestId: rid(),
    }),
  );
  if (selected.drawdyElementIds.length === 0) return [];
  const elements = unwrap(
    await issue({
      type: "command:scene:get-drawdy-elements",
      driverId,
      requestId: rid(),
      req: {
        properties: ["type"],
        drawdyElementIds: selected.drawdyElementIds,
      },
    }),
  );
  return elements.drawdyElements
    .filter((el) => el.type === "image")
    .map((el) => el.id);
}

function ensureSession(): Promise<InferenceSession> {
  if (sessionPromise) return sessionPromise;
  const created = createSession();
  sessionPromise = created;
  created.catch(() => {
    if (sessionPromise === created) sessionPromise = null;
  });
  return created;
}

async function createSession(): Promise<InferenceSession> {
  loadOrt();
  const bytes = await download(MODEL.url, (percent) =>
    toast(`Loading model (${MODEL.sizeMb} MB)… ${percent}%`),
  );
  await toast("Verifying model…");
  await verify(bytes, MODEL.sha256);
  await toast("Preparing model…");
  return ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

function loadOrt(): void {
  if (ortLoaded) return;
  importScripts(ORT_SCRIPT);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1;
  ortLoaded = true;
}

async function download(
  url: string,
  onProgress: (percent: number) => Promise<void>,
): Promise<Uint8Array<ArrayBuffer>> {
  await onProgress(0);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`model download failed (HTTP ${res.status})`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || total === 0) {
    return new Uint8Array(await res.arrayBuffer());
  }
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let received = 0;
  let lastReported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > total) {
      throw new Error("model download exceeded its declared size");
    }
    out.set(value, received);
    received += value.length;
    const percent = Math.floor((received / total) * 100);
    if (percent >= lastReported + 5) {
      lastReported = percent;
      await onProgress(percent);
    }
  }
  if (received !== total) {
    throw new Error("model download ended early");
  }
  return out;
}

async function verify(
  bytes: Uint8Array<ArrayBuffer>,
  expectedSha256: string,
): Promise<void> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return;
  const digest = await subtle.digest("SHA-256", bytes);
  if (bytesToHex(digest) !== expectedSha256) {
    throw new Error("model integrity check failed");
  }
}

async function processImage(
  session: InferenceSession,
  drawdyElementId: string,
): Promise<void> {
  const source = unwrap(
    await issue({
      type: "command:scene:get-image-source",
      driverId,
      requestId: rid(),
      req: { drawdyElementId },
    }),
  );
  const bitmap = await createImageBitmap(source.blob);
  const natural: Size = { width: bitmap.width, height: bitmap.height };
  const target = fitWithin(natural, MAX_SIDE);
  const modelSize: Size = { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE };
  const full = rasterize(bitmap, target);
  const small = rasterize(bitmap, modelSize);
  bitmap.close();

  const input = new ort.Tensor("float32", preprocess(small.data), [
    1,
    3,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  ]);
  const outputs = await session.run({ [MODEL_INPUT_NAME]: input });
  const first = outputs[session.outputNames[0]];
  const mask = maskFromOutput(first.data as Float32Array);
  const alpha = resizeMaskBilinear(mask, modelSize, target);
  applyAlpha(full.data, alpha);

  const canvas = new OffscreenCanvas(target.width, target.height);
  canvas.getContext("2d")!.putImageData(full, 0, 0);
  const png = await canvas.convertToBlob({ type: "image/png" });

  const rects = unwrap(
    await issue({
      type: "command:scene:element-rects",
      driverId,
      requestId: rid(),
      req: { drawdyElementIds: [drawdyElementId] },
    }),
  );
  const rect = rects.rects[0]?.rect;
  if (!rect) throw new Error("source image disappeared");
  unwrap(
    await issue({
      type: "command:scene:add-drawdy-elements",
      driverId,
      requestId: rid(),
      req: {
        elements: [
          {
            type: "image",
            drawdyElementId: `${drawdyElementId}:${generateId()}:no-background`,
            x: rect.x + rect.width + RESULT_GAP,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            blob: png,
          },
        ],
      },
    }),
  );
}

function rasterize(bitmap: ImageBitmap, size: Size): ImageData {
  const canvas = new OffscreenCanvas(size.width, size.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, size.width, size.height);
  return ctx.getImageData(0, 0, size.width, size.height);
}

async function toast(text: string, autoHideMs?: number): Promise<void> {
  const token = ++toastToken;
  const size = unwrap(
    await issue({
      type: "command:dom:window-size",
      driverId,
      requestId: rid(),
    }),
  );
  const position = {
    x: Math.round(size.width / 2 - TOAST_WIDTH / 2),
    y: 16,
  };
  const schema = toastSchema(text);
  await issue({
    type: "command:dom:upsert-floating-element",
    driverId,
    requestId: rid(),
    req: { domId: TOAST_ID, position, asPopover: false, schema },
  });
  if (autoHideMs !== undefined) {
    setTimeout(() => {
      if (token !== toastToken) return;
      void issue({
        type: "command:dom:remove-floating-element",
        driverId,
        requestId: rid(),
        req: { domId: TOAST_ID },
      });
    }, autoHideMs);
  }
}

function toastSchema(text: string): DomElementSchema {
  return {
    type: "row",
    styles: {
      width: [TOAST_WIDTH, "px"],
      padding: [12, "px"],
      backgroundColor: styling.surface,
      borderColor: styling.border,
      borderWidth: [1, "px"],
      borderRadius: [10, "px"],
      mainAxisAlignment: "center",
      crossAxisAlignment: "center",
      pointerEvents: "none",
    },
    children: [
      {
        type: "text",
        child: text,
        styles: {
          color: styling.foreground,
          fontSize: [13, "px"],
          fontWeight: "medium",
          textAlign: "center",
        },
      },
    ],
  };
}

type Envelope = {
  res:
    | { error?: never; value: unknown }
    | { error: ProtocolCommandError; value?: never };
};

function unwrap<T extends Envelope>(
  res: T,
): Extract<T["res"], { error?: never }>["value"] {
  if (res.res.error !== undefined) {
    throw new Error(res.res.error.message ?? res.res.error.type);
  }
  return res.res.value as Extract<T["res"], { error?: never }>["value"];
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
