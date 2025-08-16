import os
import cv2
import json
import argparse
import numpy as np
import torch
from doclayout_yolo import YOLOv10


def normalize(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


ALIASES = {
    "text": "plain_text",
    "plain text": "plain_text",
    "plain-text": "plain_text",
    "figure-caption": "figure_caption",
    "table-caption": "table_caption",
    "table-footnote": "table_footnote",
    "formula-caption": "formula_caption",
    "isolated_formula": "isolate_formula",
    "isolated-formula": "isolate_formula",
    "isolated formula": "isolate_formula",
}


def map_alias(token: str) -> str:
    t = token.strip().lower()
    return ALIASES.get(t, token)


def parse_keep_classes(names_dict, keep_arg: str):
    # names_dict: id -> class name
    norm_model = {k: normalize(v) for k, v in names_dict.items()}
    inv = {}
    for k, v in norm_model.items():
        inv.setdefault(v, []).append(k)

    tokens = [t for t in keep_arg.split(",") if t.strip()]
    ids, missing = [], []
    for raw in tokens:
        aliased = map_alias(raw)
        key = normalize(aliased)
        if key in inv:
            ids.extend(inv[key])
        else:
            missing.append(raw.strip())

    if missing:
        raise SystemExit(
            f"Requested classes not found: {missing}. "
            f"Available: {sorted(set(names_dict.values()))}"
        )
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--image-path", required=True, type=str, help="File path or glob pattern")
    parser.add_argument("--res-path", default="outputs", type=str)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--conf", default=0.35, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--line-width", default=4, type=int)
    parser.add_argument("--font-size", default=18, type=int)
    parser.add_argument("--keep-class", default="abandon", type=str,
                        help="Class name or comma separated names. Example: abandon")
    parser.add_argument("--save-json", action="store_true", help="Also save per-image JSON with kept boxes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.res_path, exist_ok=True)

    model = YOLOv10(args.model)
    det_res = model.predict(
        args.image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )

    if not det_res:
        print("No images processed.")
        return

    # Determine class ids to keep based on first result
    first_names = det_res[0].names
    keep_ids = parse_keep_classes(first_names, args.keep_class)
    keep_names = [first_names[i] for i in keep_ids]
    print(f"Keeping classes: {keep_names}")

    total_kept = 0
    for idx, res in enumerate(det_res):
        src_path = getattr(res, "path", None) or args.image_path
        base = os.path.splitext(os.path.basename(src_path))[0]
        ext = os.path.splitext(os.path.basename(src_path))[1] or ".jpg"

        annotated_path = os.path.join(args.res_path, f"{base}_res{ext}")
        kept_only_path = os.path.join(args.res_path, f"{base}_kept_only{ext}")
        json_path = os.path.join(args.res_path, f"{base}_kept_boxes.json")

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            print(f"[{base}] No detections")
            continue

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        img = cv2.imread(src_path)
        if img is None:
            print(f"[{base}] Failed to read image at {src_path}")
            continue

        # Save annotated
        annotated_bgr = res.plot(pil=False, line_width=args.line_width, font_size=args.font_size)
        cv2.imwrite(annotated_path, annotated_bgr)

        # Flat crops in outputs folder
        kept = []
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        per_image_counter = {}  # class_name_normalized -> count

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if cls[i] in keep_ids:
                x1g = max(0, x1)
                y1g = max(0, y1)
                x2g = min(img.shape[1], x2)
                y2g = min(img.shape[0], y2)
                if x2g <= x1g or y2g <= y1g:
                    continue

                class_label = res.names[int(cls[i])]
                class_norm = normalize(class_label)

                # increment counter for this image and class
                per_image_counter.setdefault(class_norm, 0)
                per_image_counter[class_norm] += 1
                num = per_image_counter[class_norm]

                # crop_filename_abandon_01.JPG pattern
                crop_name = f"crop_{base}_{class_norm}_{num:02d}{ext}"
                crop_path = os.path.join(args.res_path, crop_name)

                crop = img[y1g:y2g, x1g:x2g]
                cv2.imwrite(crop_path, crop)
                mask[y1g:y2g, x1g:x2g] = 255

                kept.append({
                    "cls_id": int(cls[i]),
                    "cls_name": class_label,
                    "conf": float(conf[i]),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_path": crop_path.replace("\\", "/"),
                })

        kept_only = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(kept_only_path, kept_only)

        if args.save_json:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": src_path,
                        "kept_class_ids": keep_ids,
                        "kept_class_names": [res.names[i] for i in keep_ids],
                        "boxes": kept,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        print(f"[{base}] Annotated: {annotated_path}")
        print(f"[{base}] Kept only page: {kept_only_path}")
        if args.save_json:
            print(f"[{base}] JSON: {json_path}")
        print(f"[{base}] Saved crops: {sum(1 for k in kept if normalize(k['cls_name']) in [normalize(n) for n in keep_names])}")
        total_kept += len(kept)

    print(f"Done. Total kept boxes: {total_kept}")


if __name__ == "__main__":
    main()
