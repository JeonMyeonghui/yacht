# main_opencv_demo.py
# -*- coding: utf-8 -*-
"""
OpenCV 핵심 데모 (영상 내용 기반)
① 이미지 출력/ROI/도형
② 카메라·동영상 루프
③ 얼굴 검출 + 캐릭터 PNG 오버레이
④ 반자동 문서 스캐너(사각 외곽 → 투시보정)
⑤ 카드(명함/카드) 다중 검출 → 개별 저장
"""

import os
import cv2
import numpy as np
import argparse

# -----------------------------
# 공용 유틸 (점 정렬, 투시 보정)
# -----------------------------
def order_points(pts):
    """사각형 꼭짓점 4개를 (좌상, 우상, 우하, 좌하) 순서로 정렬"""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)                 # 좌상(최소), 우하(최대)
    diff = np.diff(pts, axis=1)         # 우상(최소), 좌하(최대)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def four_point_transform(image, pts):
    """4점(사각형) 기반 투시 보정"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 변 길이 추정 → 출력 폭/높이 결정
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))

    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# -----------------------------
# ① 이미지 출력 / ROI / 도형 그리기
# -----------------------------
def demo_image_draw(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {img_path}")

    h, w = img.shape[:2]

    # ROI(슬라이싱)로 특정 영역 색 채우기 (BGR)
    roi_y1, roi_y2 = int(h*0.15), int(h*0.35)
    roi_x1, roi_x2 = int(w*0.15), int(w*0.40)
    img[roi_y1:roi_y2, roi_x1:roi_x2] = (0, 255, 255)  # 노랑

    # 직선 (안티에일리어싱)
    cv2.line(img, (20, 20), (w-20, 20), (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 사각형(외곽선/채우기)
    cv2.rectangle(img, (int(w*0.60), int(h*0.15)), (int(w*0.90), int(h*0.35)), (0, 0, 255), 3)
    cv2.rectangle(img, (int(w*0.60), int(h*0.40)), (int(w*0.90), int(h*0.60)), (0, 128, 0), -1)  # 채우기

    # 원
    cv2.circle(img, (int(w*0.20), int(h*0.70)), 60, (255, 0, 0), 3)

    # 다각형(폴리라인) & 채우기
    pts = np.array([
        [int(w*0.10), int(h*0.85)],
        [int(w*0.25), int(h*0.75)],
        [int(w*0.35), int(h*0.90)]
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [pts + np.array([0, -50])], (128, 0, 128))  # 위쪽에 채운 삼각형

    # 텍스트
    cv2.putText(img, "OpenCV Demo", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Image Draw Demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# ② 카메라/동영상 루프(표준 패턴)
# -----------------------------
def demo_video_loop(source):
    """
    source: 0(기본 카메라) 또는 'video.mp4'와 같은 파일 경로
    종료: 키보드 q
    """
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
# ③ 얼굴 검출 + 캐릭터 PNG 오버레이
# -----------------------------
def overlay_transparent(bg, overlay_rgba, x, y, w, h):
    """
    bg(BGR)에 overlay_rgba(RGBA)를 (x,y) 좌상단에 w*h 크기로 알파블렌딩
    """
    ov = cv2.resize(overlay_rgba, (w, h), interpolation=cv2.INTER_AREA)
    if ov.shape[2] == 3:
        # 알파 채널 없음 → 전체 불투명 처리
        alpha = np.ones((h, w), dtype=float)
        ov_bgr = ov
    else:
        alpha = ov[:, :, 3] / 255.0
        ov_bgr = ov[:, :, :3]

    # 배경 ROI
    y1, y2 = max(0, y), min(bg.shape[0], y + h)
    x1, x2 = max(0, x), min(bg.shape[1], x + w)
    ov_y1, ov_y2 = y1 - y, y1 - y + (y2 - y1)
    ov_x1, ov_x2 = x1 - x, x1 - x + (x2 - x1)

    if y1 >= y2 or x1 >= x2:
        return bg  # 화면 밖이면 무시

    roi = bg[y1:y2, x1:x2]
    alpha_roi = alpha[ov_y1:ov_y2, ov_x1:ov_x2][:, :, None]  # (h,w,1)

    blended = (alpha_roi * ov_bgr[ov_y1:ov_y2, ov_x1:ov_x2] +
               (1 - alpha_roi) * roi).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended
    return bg

def face_overlay_video(source, overlay_png, scale=1.2):
    """
    얼굴 검출(HaarCascade) 후 캐릭터 PNG를 얼굴 박스보다 약간 크게 오버레이
    종료: q
    """
    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    if face_cascade.empty():
        raise RuntimeError("HaarCascade 로드 실패")

    ov = cv2.imread(overlay_png, cv2.IMREAD_UNCHANGED)  # RGBA 필요
    if ov is None:
        raise FileNotFoundError(f"오버레이 PNG 불러오기 실패: {overlay_png}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("더 이상 가져올 프레임이 없습니다.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # 얼굴 박스보다 scale 배 확대하여 자연스럽게 덮기
            W = int(w * scale)
            H = int(h * scale)
            X = x - (W - w)//2
            Y = y - int(H*0.35)  # 살짝 위로 올려 모자/귀 등 씌우기 느낌
            frame = overlay_transparent(frame, ov, X, Y, W, H)

        cv2.imshow("Face + Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# ④ 반자동 문서 스캐너
#    - 가장 큰 사각 외곽을 찾아 투시 보정
# -----------------------------
def scan_document(image_path, out_path=None, show_steps=False):
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    img = orig.copy()
    ratio = 800.0 / img.shape[1]  # 가로 800 기준 리사이즈
    img = cv2.resize(img, (800, int(img.shape[0]*ratio)), interpolation=cv2.INTER_AREA)

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
            doc_quad = approx.reshape(4, 2)
            break

    if doc_quad is None:
        raise RuntimeError("사각 외곽을 찾지 못했습니다. 배경 대비/프레이밍을 조정해보세요.")

    # 원본 좌표로 역환산
    doc_quad = (doc_quad / ratio).astype(np.float32)
    warped = four_point_transform(orig, doc_quad)

    # 스캐너 느낌(흑백+명암강조)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    if out_path:
        cv2.imwrite(out_path, scanned)

    if show_steps:
        cv2.imshow("edges", edges)
        cv2.imshow("warped", warped)
        cv2.imshow("scanned", scanned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scanned

# -----------------------------
# ⑤ 카드(명함/카드) 다중 검출 → 개별 저장
#    - 여러 사각형을 찾아 각자 투시보정 후 저장
# -----------------------------
def split_cards(image_path, out_dir="cards_out", min_area=8_000):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")

    draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
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

        quad = approx.reshape(4, 2).astype(np.float32)
        warped = four_point_transform(img, quad)

        # 카드 출력을 일정 크기로 정규화(예: 480x300)
        card_norm = cv2.resize(warped, (480, 300), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(out_dir, f"card_{idx:02d}.png")
        cv2.imwrite(save_path, card_norm)
        idx += 1

        cv2.polylines(draw, [approx], True, (0, 255, 0), 2)

    cv2.imshow("Detected Cards", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"총 {idx-1}개 저장됨 → {os.path.abspath(out_dir)}")

# -----------------------------
# 실행 엔트리
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 핵심 데모")
    p.add_argument("--mode", required=True,
                   choices=["image_draw", "video", "face", "scan", "cards"],
                   help="실행 모드 선택")
    p.add_argument("--image", help="이미지 경로 (image_draw/scan/cards에서 사용)")
    p.add_argument("--source", default="0",
                   help="영상 소스: 0(기본 카메라) 또는 동영상 파일 경로 (video/face)")
    p.add_argument("--overlay", help="캐릭터 PNG 경로(face 모드에서 사용)")
    p.add_argument("--out", help="scan 결과 저장 경로")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "image_draw":
        if not args.image:
            raise ValueError("--image 를 지정하십시오.")
        demo_image_draw(args.image)

    elif args.mode == "video":
        src = 0 if args.source == "0" else args.source
        demo_video_loop(src)

    elif args.mode == "face":
        if not args.overlay:
            raise ValueError("--overlay 캐릭터 PNG 경로를 지정하십시오.")
        src = 0 if args.source == "0" else args.source
        face_overlay_video(src, args.overlay, scale=1.2)

    elif args.mode == "scan":
        if not args.image:
            raise ValueError("--image 를 지정하십시오.")
        scan_document(args.image, out_path=args.out or "scanned.png", show_steps=True)

    elif args.mode == "cards":
        if not args.image:
            raise ValueError("--image 를 지정하십시오.")
        split_cards(args.image, out_dir="cards_out")

if __name__ == "__main__":
    main()
