# opencv_advanced_demo.py
# -*- coding: utf-8 -*-
"""
OpenCV 고급 데모 (이전 plus/확장 버전 통합)

▶ 이번 수정 사항(\u2192 SystemExit: 2 방지)
- 인자가 전혀 없는 실행 시 `--mode help`로 자동 대체 → 친절한 사용법/예시를 출력하고 정상 종료(코드 0)
- `--mode`의 기본값을 `help`로 두어, 필수 인자 누락으로 인한 `SystemExit: 2`를 피함
- 잘못된/누락 인자는 **친절한 오류 메시지 + 해당 모드 예시**를 출력
- `--mode selftest`를 추가하여 **GUI/카메라 없이도 동작 확인 가능한 단위 테스트** 제공

필수:
    pip install opencv-python numpy mediapipe

빠른 실행 예:
    python opencv_advanced_demo.py                         # (도움말 자동 표시)
    python opencv_advanced_demo.py --mode help             # 도움말/예시
    python opencv_advanced_demo.py --mode selftest         # 단위 테스트 실행(창/카메라 없음)
    python opencv_advanced_demo.py --mode face_landmark --overlay character.png --source 0
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
import sys
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
# 추가 기능: 다중 얼굴 캐릭터, HSV 일괄 적용, 실시간 스캔 저장
# -----------------------------

def _is_image_file(p):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    return os.path.isfile(p) and p.lower().endswith(exts)


def load_overlays_from_dir(dir_path):
    files = sorted([f for f in os.listdir(dir_path)
                    if f.lower().endswith(('.png', '.webp'))])
    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(dir_path, f), cv2.IMREAD_UNCHANGED)
        if im is not None:
            imgs.append(im)
    if not imgs:
        raise FileNotFoundError("오버레이 이미지가 없습니다(디렉터리 확인)")
    return imgs


def face_landmark_overlay_multi(source, overlay_pngs=None, overlay_dir=None,
                                eye_scale=2.2, y_offset_ratio=0.55):
    """여러 캐릭터 PNG를 얼굴마다 다르게 씌우기(좌→우 순서로 순환 배정).
    overlay_pngs: 리스트 경로, overlay_dir: 디렉터리 경로 중 하나 사용
    """
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection

    # 오버레이 이미지들 준비
    overlays = []
    if overlay_dir:
        overlays = load_overlays_from_dir(overlay_dir)
    if overlay_pngs:
        for p in overlay_pngs:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is not None:
                overlays.append(im)
    if not overlays:
        raise ValueError("오버레이 PNG를 --overlay 또는 --overlays 또는 --overlay_dir 로 지정하십시오.")

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
            res = fd.process(rgb)

            if res.detections:
                # 좌→우(x) 순으로 정렬
                dets = sorted(res.detections, key=lambda d: d.location_data.relative_keypoints[0].x)
                for i, det in enumerate(dets):
                    kp = det.location_data.relative_keypoints
                    rx, ry = int(kp[0].x * w), int(kp[0].y * h)
                    lx, ly = int(kp[1].x * w), int(kp[1].y * h)
                    nx, ny = int(kp[2].x * w), int(kp[2].y * h)
                    cx, cy = (rx + lx) // 2, (ry + ly) // 2
                    eye_dist = max(1, int(math.hypot(lx - rx, ly - ry)))
                    angle = math.degrees(math.atan2(ly - ry, lx - rx))
                    overlay_w = int(eye_dist * eye_scale)
                    overlay_h = overlay_w
                    y_offset = int((cy - ny) * y_offset_ratio)
                    X = cx - overlay_w // 2
                    Y = cy - overlay_h // 2 - y_offset
                    ov = rotate_rgba(overlays[i % len(overlays)], angle)
                    frame = overlay_transparent(frame, ov, X, Y, overlay_w, overlay_h)

            cv2.imshow("Face Landmark Overlay (Multi)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release(); cv2.destroyAllWindows()


# HSV 파라미터 적용 유틸

def hsv_mask_from_params(img_bgr, params):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hmin, Hmax = int(params['Hmin']), int(params['Hmax'])
    Smin, Smax = int(params['Smin']), int(params['Smax'])
    Vmin, Vmax = int(params['Vmin']), int(params['Vmax'])
    k = int(params.get('Kernel', 3))
    k = k if k % 2 == 1 else k + 1
    open_it = int(params.get('Open', 0))
    close_it = int(params.get('Close', 0))

    mask = cv2.inRange(hsv, (Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if open_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_it)
    if close_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_it)
    return mask


def hsv_apply(image_or_dir, params_json, out_dir="hsv_out", save_mask=True):
    os.makedirs(out_dir, exist_ok=True)
    with open(params_json, 'r', encoding='utf-8') as f:
        params = json.load(f)

    paths = []
    if os.path.isdir(image_or_dir):
        for name in sorted(os.listdir(image_or_dir)):
            p = os.path.join(image_or_dir, name)
            if _is_image_file(p):
                paths.append(p)
    else:
        paths = [image_or_dir]

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[건너뜀] 이미지 로드 실패: {p}"); continue
        mask = hsv_mask_from_params(img, params)
        masked = cv2.bitwise_and(img, img, mask=mask)
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_masked.png"), masked)
        if save_mask:
            cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask)
    print(f"완료: {len(paths)}개 처리 → {os.path.abspath(out_dir)}")


# HSV 전처리 기반 카드 분할(선택적)

def split_cards_with_hsv(image_path, params_json, out_dir="cards_out_hsv", min_area=8000):
    with open(params_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")
    mask = hsv_mask_from_params(img, params)
    masked = cv2.bitwise_and(img, img, mask=mask)

    draw = masked.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        warped = four_point_transform(img, quad)
        card_norm = cv2.resize(warped, (480, 300), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(out_dir, f"card_{idx:02d}.png")
        cv2.imwrite(save_path, card_norm); idx += 1
        cv2.polylines(draw, [approx], True, (0, 255, 0), 2)

    cv2.imshow("Detected Cards (HSV)", draw)
    cv2.waitKey(0); cv2.destroyAllWindows()
    print(f"총 {idx-1}개 저장됨 → {os.path.abspath(out_dir)}")


# 실시간 웹캠 문서 스캔 자동 저장

def scan_cam_auto(source=0, out_dir="scans", interval_sec=2.0, stable_frames=10,
                   move_thresh_ratio=0.02, area_thresh_ratio=0.15):
    """웹캠에서 사각 문서를 검출해 안정적으로 유지되면 자동 저장.
    - interval_sec: 저장 최소 간격(초)
    - stable_frames: 연속 안정 프레임 수
    - move_thresh_ratio: 중심 이동 허용 비율(대각선 대비)
    - area_thresh_ratio: 면적 변화 허용 비율
    키: q 종료, s 수동 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패: {source}")

    import time
    last_save = 0.0
    stable = 0
    prev_center, prev_area = None, None

    def detect_quad(frame):
        ratio = 800.0 / frame.shape[1]
        img = cv2.resize(frame, (800, int(frame.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2)
                return quad, ratio, edges
        return None, ratio, edges

    def save_scan(frame, quad, ratio):
        quad_full = (quad / ratio).astype(np.float32)
        warped = four_point_transform(frame, quad_full)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 10)
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f"scan_{ts}.png")
        cv2.imwrite(path, scanned)
        print(f"[저장] {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 종료"); break

        quad, ratio, edges = detect_quad(frame)
        display = frame.copy()
        h, w = display.shape[:2]
        diag = math.hypot(w, h)

        if quad is not None:
            # 그리기(축소 좌표를 원본으로 스케일링)
            quad_full = (quad / ratio).astype(np.int32)
            cv2.polylines(display, [quad_full], True, (0, 255, 0), 2)
            # 안정성 평가
            center = tuple(np.mean(quad_full, axis=0))
            area = cv2.contourArea(quad_full)
            ok = True
            if prev_center is not None:
                move = math.hypot(center[0]-prev_center[0], center[1]-prev_center[1])
                if move > move_thresh_ratio * diag:
                    ok = False
            if prev_area is not None:
                if abs(area - prev_area) / max(1.0, prev_area) > area_thresh_ratio:
                    ok = False
            if ok:
                stable += 1
            else:
                stable = 0
            prev_center, prev_area = center, area

            # 상태 텍스트
            cv2.putText(display, f"stable: {stable}/{stable_frames}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if stable>=stable_frames else (0,200,200), 2, cv2.LINE_AA)

            # 자동 저장
            now = time.time()
            if stable >= stable_frames and (now - last_save) >= float(interval_sec):
                save_scan(frame, quad, ratio)
                last_save = now
                stable = 0
        else:
            stable = 0; prev_center = None; prev_area = None
            cv2.putText(display, "문서 미검출", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('ScanCam Auto', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and quad is not None:
            save_scan(frame, quad, ratio)

    cap.release(); cv2.destroyAllWindows()


# -----------------------------
# CLI + 도움말/예시 + SelfTest
# -----------------------------

def print_examples(parser):
    print("\n사용 예시:")
    print("  python opencv_advanced_demo.py                         # 도움말 자동 표시")
    print("  python opencv_advanced_demo.py --mode help             # 도움말/예시")
    print("  python opencv_advanced_demo.py --mode selftest         # 단위 테스트(창/카메라 없음)")
    print("  python opencv_advanced_demo.py --mode face_landmark --overlay character.png --source 0")
    print("  python opencv_advanced_demo.py --mode hsv_tuner --image cards.jpg --save hsv_params.json")
    print("  python opencv_advanced_demo.py --mode pipeline_tuner --image doc.jpg")
    print("  python opencv_advanced_demo.py --mode scan --image doc.jpg --out scanned.png")
    print("  python opencv_advanced_demo.py --mode cards_auto --image cards.jpg\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OpenCV 고급 데모 (확장)", add_help=True)
    p.add_argument("--mode", default="help",
                   choices=[
                       "help",              # 도움말/예시 출력 후 종료
                       "selftest",          # 단위 테스트 실행
                       "face_landmark",       # 미디어파이프 기반 정밀 오버레이(단일)
                       "face_landmark_multi", # 미디어파이프 + 다중 캐릭터
                       "hsv_tuner",           # HSV 튜너(저장)
                       "hsv_apply",           # 저장한 HSV 파라미터 적용(파일/폴더)
                       "pipeline_tuner",      # 전처리 파이프라인 튜너
                       "scan",                # 문서 스캔(정지 이미지)
                       "scan_cam_auto",       # 실시간 문서 스캔 자동 저장
                       "cards_auto",          # 카드 자동 분할(단일)
                       "cards_auto_dir",      # 카드 자동 분할(디렉터리)
                       "video"                # 단순 영상 루프
                   ], help="실행 모드")
    p.add_argument("--source", default="0", help="0 또는 동영상 파일 경로(카메라/영상 모드)")
    p.add_argument("--image", help="입력 이미지 경로(hsv_tuner/pipeline_tuner/scan/cards_auto)")
    p.add_argument("--indir", help="입력 이미지 폴더(hsv_apply/cards_auto_dir)")
    p.add_argument("--out", help="scan 결과 저장 경로")
    p.add_argument("--outdir", help="결과 저장 폴더(hsv_apply/cards_auto_dir)")
    p.add_argument("--overlay", help="캐릭터 PNG 경로(face_landmark)")
    p.add_argument("--overlays", help="쉼표(,)로 구분한 여러 PNG 경로(face_landmark_multi)")
    p.add_argument("--overlay_dir", help="여러 PNG가 있는 디렉터리(face_landmark_multi)")
    p.add_argument("--save", help="튜너 파라미터 저장 경로(JSON)")
    p.add_argument("--params", help="HSV 파라미터(JSON) 경로(hsv_apply/split_cards_with_hsv)")
    p.add_argument("--interval", type=float, default=2.0, help="scan_cam_auto 저장 간격(초)")
    p.add_argument("--stable", type=int, default=10, help="scan_cam_auto 안정 프레임 수")

    # argparse는 인자 오류 시 SystemExit을 발생시킵니다. 전역에서 잡아 안내합니다.
    args = p.parse_args(argv)
    # 인자 전혀 없음 → help 모드 강제
    if argv is None and len(sys.argv) <= 1:
        args.mode = "help"
    return args, p


# -----------------------------
# 단위 테스트(창/카메라 없음) — ALWAYS ADD TESTS
# -----------------------------

def _test_order_points():
    pts = np.array([[100,400],[500,420],[120,120],[520,140]], dtype=np.float32)  # 임의 순서
    orded = order_points(pts)
    # 좌상은 x+y 최소, 우하는 최대
    assert np.argmin((pts[:,0]+pts[:,1])) in [np.where((pts==orded[0]).all(axis=1))[0][0]], "order_points 좌상 실패"
    assert np.argmax((pts[:,0]+pts[:,1])) in [np.where((pts==orded[2]).all(axis=1))[0][0]], "order_points 우하 실패"


def _test_rotate_overlay_and_blend():
    bg = np.zeros((120,120,3), dtype=np.uint8)
    ov = np.zeros((30,30,4), dtype=np.uint8)
    ov[:,:] = [0,255,0,255]  # 불투명 녹색 정사각
    rot = rotate_rgba(ov, 33)
    out = overlay_transparent(bg.copy(), rot, 10, 15, 60, 60)
    assert out.shape == bg.shape and out.sum() > 0, "overlay/blend 실패"


def _test_hsv_mask():
    # 파란색 배경 위 빨간 점: 파란색을 탐지하도록 범위 설정
    img = np.zeros((50,50,3), dtype=np.uint8)
    img[:] = (255,0,0)      # BGR: 파랑
    img[20:30,20:30] = (0,0,255)  # 빨강(무시되어야 함)
    params = {"Hmin":100,"Hmax":140,"Smin":50,"Smax":255,"Vmin":50,"Vmax":255,"Kernel":3,"Open":0,"Close":0}
    mask = hsv_mask_from_params(img, params)
    # 파란 배경 대부분이 검출되어야 함
    ratio = mask.mean()/255.0
    assert ratio > 0.5, f"HSV 마스크 비정상: ratio={ratio:.2f}"


def run_self_tests():
    _test_order_points()
    _test_rotate_overlay_and_blend()
    _test_hsv_mask()
    print("[selftest] 모든 테스트 통과 ✔")


def main(argv=None):
    try:
        args, parser = parse_args(argv)
    except SystemExit as e:
        # argparse 내부 종료 — 사용 예시를 함께 안내
        print("\n인자 해석 오류가 발생했습니다.")
        print_examples(None)
        raise

    if args.mode == "help":
        # 도움말 + 예시 출력 후 정상 종료
        print("OpenCV 고급 데모 — 도움말")
        print_examples(parser)
        return

    if args.mode == "selftest":
        run_self_tests()
        return

    # 이하 각 모드 실행 및 필수 인자 검증
    if args.mode == "face_landmark":
        if not args.overlay:
            print("[오류] --overlay PNG 경로가 필요합니다.")
            print("예) python opencv_advanced_demo.py --mode face_landmark --overlay character.png --source 0")
            return
        src = 0 if args.source == "0" else args.source
        face_landmark_overlay(src, args.overlay)

    elif args.mode == "face_landmark_multi":
        src = 0 if args.source == "0" else args.source
        ov_list = None
        if args.overlays:
            ov_list = [p.strip() for p in args.overlays.split(',') if p.strip()]
        if not ov_list and not args.overlay_dir:
            print("[오류] --overlays 또는 --overlay_dir 중 하나를 지정하세요.")
            print("예) --overlay_dir overlays/  또는  --overlays a.png,b.png")
            return
        face_landmark_overlay_multi(src, overlay_pngs=ov_list, overlay_dir=args.overlay_dir)

    elif args.mode == "hsv_tuner":
        if not args.image:
            print("[오류] --image 경로가 필요합니다. 예) --image cards.jpg")
            return
        hsv_tuner(args.image, save_path=args.save)

    elif args.mode == "hsv_apply":
        target = args.indir or args.image
        if not target or not args.params:
            print("[오류] --image 또는 --indir, 그리고 --params(JSON)가 필요합니다.")
            print("예) --indir samples/ --params hsv_params.json")
            return
        outdir = args.outdir or "hsv_out"
        hsv_apply(target, args.params, out_dir=outdir, save_mask=True)

    elif args.mode == "pipeline_tuner":
        if not args.image:
            print("[오류] --image 경로가 필요합니다. 예) --image doc.jpg")
            return
        pipeline_tuner(args.image)

    elif args.mode == "scan":
        if not args.image:
            print("[오류] --image 경로가 필요합니다. 예) --image doc.jpg")
            return
        scan_document(args.image, out_path=args.out or "scanned.png", show_steps=True)

    elif args.mode == "scan_cam_auto":
        src = 0 if args.source == "0" else args.source
        scan_cam_auto(src, out_dir=args.outdir or "scans", interval_sec=args.interval, stable_frames=args.stable)

    elif args.mode == "cards_auto":
        if not args.image:
            print("[오류] --image 경로가 필요합니다. 예) --image cards.jpg")
            return
        split_cards(args.image, out_dir=args.outdir or "cards_out")

    elif args.mode == "cards_auto_dir":
        if not args.indir:
            print("[오류] --indir 폴더가 필요합니다. 예) --indir cards_batch/")
            return
        outdir = args.outdir or "cards_out"
        os.makedirs(outdir, exist_ok=True)
        count = 0
        for name in sorted(os.listdir(args.indir)):
            p = os.path.join(args.indir, name)
            if _is_image_file(p):
                base = os.path.splitext(os.path.basename(p))[0]
                sub = os.path.join(outdir, base)
                os.makedirs(sub, exist_ok=True)
                try:
                    split_cards(p, out_dir=sub)
                    count += 1
                except Exception as e:
                    print(f"[오류] {p}: {e}")
        print(f"총 {count}개 이미지를 처리했습니다 → {os.path.abspath(outdir)}")

    elif args.mode == "video":
        src = 0 if args.source == "0" else args.source
        video_loop(src)


if __name__ == "__main__":
    # argparse의 SystemExit를 잡아 사용자에게 더 친절한 메시지를 보여줌
    try:
        main()
    except SystemExit as e:
        # 잘못된 인자 등으로 종료된 경우, 코드 2 유지하되 예시를 이미 출력함
        if e.code == 2:
            print("\n[힌트] 필요한 인자를 확인하려면:  python opencv_advanced_demo.py --mode help\n")
        raise


