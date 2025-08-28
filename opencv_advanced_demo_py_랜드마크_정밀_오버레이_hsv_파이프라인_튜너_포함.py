# opencv_advanced_demo.py
# -*- coding: utf-8 -*-
"""
OpenCV 고급 데모 (이전 plus 버전 확장)

신규 기능 요약
- 얼굴 랜드마크(미디어파이프 FaceDetection) 기반 정밀 배치 + 각도/스케일 자동 보정 오버레이
- HSV 마스크 실시간 튜너(범위/커널/모폴로지 저장 가능)
- 문서/카드 전처리 파이프라인 실시간 튜너(블러, 캐니, 팽창/침식, 적응형 이진화)

기존 기능도 포함(문서 스캔, 카드 분할 등)하여 한 파일로 실행 가능하도록 구성.

필수:
    pip install opencv-python numpy mediapipe

실행 예:
    python opencv_advanced_demo.py --mode face_landmark --source 0 --overlay character.png
    python opencv_advanced_demo.py --mode hsv_tuner --image cards.jpg --save hsv_params.json
    python opencv_advanced_demo.py --mode pipeline_tuner --image doc.jpg
    python opencv_advanced_demo.py --mode scan --image doc.jpg --out scanned.png
    python opencv_advanced_demo.py --mode cards_auto --image cards.jpg
"""

import os
import cv2
import math
import json
import argparse
import numpy as np

# -----------------------------
# 공용: 점 정렬 & 투시보정
# -----------------------------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))
    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# -----------------------------
# 오버레이 유틸 (알파 블렌딩, 회전)
# -----------------------------
def overlay_transparent(bg, overlay_rgba, x, y, w, h):
    """bg(BGR)에 overlay_rgba(RGBA)를 (x,y) 좌상단에 w*h 크기로 알파 블렌딩."""
    ov = cv2.resize(overlay_rgba, (w, h), interpolation=cv2.INTER_AREA)
    if ov.shape[2] == 3:
        alpha = np.ones((h, w), dtype=float)
        ov_bgr = ov
    else:
        alpha = ov[:, :, 3] / 255.0
        ov_bgr = ov[:, :, :3]

    y1, y2 = max(0, y), min(bg.shape[0], y + h)
    x1, x2 = max(0, x), min(bg.shape[1], x + w)
    ov_y1, ov_y2 = y1 - y, y1 - y + (y2 - y1)
    ov_x1, ov_x2 = x1 - x, x1 - x + (x2 - x1)
    if y1 >= y2 or x1 >= x2:
        return bg

    roi = bg[y1:y2, x1:x2]
    a = alpha[ov_y1:ov_y2, ov_x1:ov_x2][:, :, None]
    blended = (a * ov_bgr[ov_y1:ov_y2, ov_x1:ov_x2] + (1 - a) * roi).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended
    return bg


def rotate_rgba(img_rgba, angle_deg, scale=1.0):
    (h, w) = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - w / 2
    M[1, 2] += (nH / 2) - h / 2
    rot = cv2.warpAffine(img_rgba, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return rot


# -----------------------------
# 얼굴 랜드마크 기반 정밀 오버레이 (MediaPipe FaceDetection)
# -----------------------------
def face_landmark_overlay(source, overlay_png, eye_scale=2.2, y_offset_ratio=0.55):
    """
    - MediaPipe FaceDetection으로 좌/우 눈 중심과 코 좌표를 얻고
      눈-눈 거리로 스케일, 눈 기울기로 각도, 눈 중점 기준으로 위치를 산정하여
      캐릭터 PNG(RGBA)를 자연스럽게 씌웁니다.
    - eye_scale: 오버레이 폭을 (양 눈 사이 거리 * eye_scale)로 산정
    - y_offset_ratio: 눈-코 거리 비율만큼 위로 올려 모자/귀 착용처럼 배치
    종료: q
    """
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection

    ov_rgba = cv2.imread(overlay_png, cv2.IMREAD_UNCHANGED)
    if ov_rgba is None:
        raise FileNotFoundError(f"오버레이 PNG 불러오기 실패: {overlay_png}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패: {source}")

    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("더 이상 가져올 프레임이 없습니다.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = fd.process(rgb)

            if result.detections:
                for det in result.detections:
                    kp = det.location_data.relative_keypoints
                    # MediaPipe: [0]=right_eye, [1]=left_eye, [2]=nose_tip
                    rx, ry = int(kp[0].x * w), int(kp[0].y * h)
                    lx, ly = int(kp[1].x * w), int(kp[1].y * h)
                    nx, ny = int(kp[2].x * w), int(kp[2].y * h)

                    # 눈 중심/거리/각도
                    cx, cy = (rx + lx) // 2, (ry + ly) // 2
                    eye_dist = max(1, int(math.hypot(lx - rx, ly - ry)))
                    angle = math.degrees(math.atan2(ly - ry, lx - rx))

                    # 스케일 및 위치 산정
                    overlay_w = int(eye_dist * eye_scale)
                    overlay_h = overlay_w  # 정사각 가정(필요 시 비율 조절)
                    # 눈-코 거리만큼 위쪽 보정
                    y_offset = int((cy - ny) * y_offset_ratio)
                    X = cx - overlay_w // 2
                    Y = cy - overlay_h // 2 - y_offset

                    ov_rot = rotate_rgba(ov_rgba, angle)
                    frame = overlay_transparent(frame, ov_rot, X, Y, overlay_w, overlay_h)

            cv2.imshow("Face Landmark Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# HSV 마스크 실시간 튜너 (저장 가능)
# -----------------------------
def hsv_tuner(image_path, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("HSV Tuner")
    # 초기값(전체 범위)
    cv2.createTrackbar("Hmin", "HSV Tuner", 0, 179, lambda x: None)
    cv2.createTrackbar("Hmax", "HSV Tuner", 179, 179, lambda x: None)
    cv2.createTrackbar("Smin", "HSV Tuner", 0, 255, lambda x: None)
    cv2.createTrackbar("Smax", "HSV Tuner", 255, 255, lambda x: None)
    cv2.createTrackbar("Vmin", "HSV Tuner", 0, 255, lambda x: None)
    cv2.createTrackbar("Vmax", "HSV Tuner", 255, 255, lambda x: None)
    cv2.createTrackbar("Kernel", "HSV Tuner", 1, 21, lambda x: None)  # 홀수 권장
    cv2.createTrackbar("Open", "HSV Tuner", 0, 5, lambda x: None)
    cv2.createTrackbar("Close", "HSV Tuner", 0, 5, lambda x: None)

    print("[HSV Tuner] q: 종료, s: 파라미터 저장")

    while True:
        Hmin = cv2.getTrackbarPos("Hmin", "HSV Tuner")
        Hmax = cv2.getTrackbarPos("Hmax", "HSV Tuner")
        Smin = cv2.getTrackbarPos("Smin", "HSV Tuner")
        Smax = cv2.getTrackbarPos("Smax", "HSV Tuner")
        Vmin = cv2.getTrackbarPos("Vmin", "HSV Tuner")
        Vmax = cv2.getTrackbarPos("Vmax", "HSV Tuner")
        k = max(1, cv2.getTrackbarPos("Kernel", "HSV Tuner"))
        if k % 2 == 0:
            k += 1
        ksize = (k, k)
        open_it = cv2.getTrackbarPos("Open", "HSV Tuner")
        close_it = cv2.getTrackbarPos("Close", "HSV Tuner")

        mask = cv2.inRange(hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        if open_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_it)
        if close_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_it)

        masked = cv2.bitwise_and(img, img, mask=mask)
        stack = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked])
        cv2.imshow("HSV Tuner", stack)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and save_path:
            params = {
                "Hmin": Hmin, "Hmax": Hmax,
                "Smin": Smin, "Smax": Smax,
                "Vmin": Vmin, "Vmax": Vmax,
                "Kernel": k, "Open": open_it, "Close": close_it,
            }
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            print(f"저장됨: {os.path.abspath(save_path)}")

    cv2.destroyAllWindows()


# -----------------------------
# 문서/카드 전처리 파이프라인 튜너
# -----------------------------
def _odd(val, minimum=3):
    val = max(minimum, int(val))
    return val if val % 2 == 1 else val + 1


def pipeline_tuner(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Pipeline Tuner")
    cv2.createTrackbar("BlurK", "Pipeline Tuner", 5, 31, lambda x: None)
    cv2.createTrackbar("Canny1", "Pipeline Tuner", 60, 255, lambda x: None)
    cv2.createTrackbar("Canny2", "Pipeline Tuner", 180, 255, lambda x: None)
    cv2.createTrackbar("Dilate", "Pipeline Tuner", 1, 5, lambda x: None)
    cv2.createTrackbar("Erode", "Pipeline Tuner", 0, 5, lambda x: None)
    cv2.createTrackbar("AdapBlk", "Pipeline Tuner", 21, 51, lambda x: None)
    cv2.createTrackbar("AdapC", "Pipeline Tuner", 10, 40, lambda x: None)  # 표시값 0..40 -> 실제 -20..20

    print("[Pipeline] q: 종료")

    while True:
        k = _odd(cv2.getTrackbarPos("BlurK", "Pipeline Tuner"), 1)
        c1 = cv2.getTrackbarPos("Canny1", "Pipeline Tuner")
        c2 = max(c1 + 1, cv2.getTrackbarPos("Canny2", "Pipeline Tuner"))
        d_it = cv2.getTrackbarPos("Dilate", "Pipeline Tuner")
        e_it = cv2.getTrackbarPos("Erode", "Pipeline Tuner")
        blk = _odd(cv2.getTrackbarPos("AdapBlk", "Pipeline Tuner"), 3)
        cval = cv2.getTrackbarPos("AdapC", "Pipeline Tuner") - 20

        blur = cv2.GaussianBlur(gray0, (k, k), 0)
        edge = cv2.Canny(blur, c1, c2)
        kernel = np.ones((3, 3), np.uint8)
        proc = edge.copy()
        if d_it > 0:
            proc = cv2.dilate(proc, kernel, iterations=d_it)
        if e_it > 0:
            proc = cv2.erode(proc, kernel, iterations=e_it)

        adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blk, cval)

        stack = np.hstack([
            cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow("Pipeline Tuner", stack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# -----------------------------
# 문서 스캔 & 카드 자동 분할 (개선형)
# -----------------------------
def scan_document(image_path, out_path=None, show_steps=False):
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    ratio = 800.0 / orig.shape[1]
    img = cv2.resize(orig, (800, int(orig.shape[0] * ratio)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # 사각형 유사도 체크(내각, 볼록성)
            if cv2.isContourConvex(approx):
                doc_quad = approx.reshape(4, 2)
                break

    if doc_quad is None:
        raise RuntimeError("사각 외곽을 찾지 못했습니다. 대비/구도를 조정하세요.")

    doc_quad = (doc_quad / ratio).astype(np.float32)
    warped = four_point_transform(orig, doc_quad)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)

    if out_path:
        cv2.imwrite(out_path, scanned)

    if show_steps:
        cv2.imshow("edges", edges)
        cv2.imshow("warped", warped)
        cv2.imshow("scanned", scanned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scanned


def split_cards(image_path, out_dir="cards_out", min_area=8000):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    draw = img.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        warped = four_point_transform(img, quad)
        card_norm = cv2.resize(warped, (480, 300), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(out_dir, f"card_{idx:02d}.png")
        cv2.imwrite(save_path, card_norm)
        idx += 1

        cv2.polylines(draw, [approx], True, (0, 255, 0), 2)

    cv2.imshow("Detected Cards", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"총 {idx - 1}개 저장됨 → {os.path.abspath(out_dir)}")


# -----------------------------
# 카메라/동영상 루프(기본)
# -----------------------------
def video_loop(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패: {source}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("더 이상 가져올 프레임이 없습니다.")
            break
        cv2.imshow("Video Loop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 고급 데모")
    p.add_argument("--mode", required=True,
                   choices=[
                       "face_landmark",    # 미디어파이프 기반 정밀 오버레이
                       "hsv_tuner",        # HSV 마스크 튜너
                       "pipeline_tuner",   # 전처리 파이프라인 튜너
                       "scan",             # 문서 스캔
                       "cards_auto",       # 카드 자동 분할
                       "video"             # 단순 영상 루프
                   ])
    p.add_argument("--source", default="0", help="0 또는 동영상 파일 경로(카메라/영상 모드)")
    p.add_argument("--image", help="입력 이미지 경로(hsv_tuner/pipeline_tuner/scan/cards_auto)")
    p.add_argument("--overlay", help="캐릭터 PNG 경로(face_landmark)")
    p.add_argument("--out", help="scan 결과 저장 경로")
    p.add_argument("--save", help="튜너 파라미터 저장 경로(JSON)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "face_landmark":
        if not args.overlay:
            raise ValueError("--overlay PNG 경로를 지정하십시오.")
        src = 0 if args.source == "0" else args.source
        face_landmark_overlay(src, args.overlay)

    elif args.mode == "hsv_tuner":
        if not args.image:
            raise ValueError("--image 경로를 지정하십시오.")
        hsv_tuner(args.image, save_path=args.save)

    elif args.mode == "pipeline_tuner":
        if not args.image:
            raise ValueError("--image 경로를 지정하십시오.")
        pipeline_tuner(args.image)

    elif args.mode == "scan":
        if not args.image:
            raise ValueError("--image 경로를 지정하십시오.")
        scan_document(args.image, out_path=args.out or "scanned.png", show_steps=True)

    elif args.mode == "cards_auto":
        if not args.image:
            raise ValueError("--image 경로를 지정하십시오.")
        split_cards(args.image, out_dir="cards_out")

    elif args.mode == "video":
        src = 0 if args.source == "0" else args.source
        video_loop(src)


if __name__ == "__main__":
    main()
