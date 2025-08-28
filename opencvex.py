# main_opencv_demo_plus.py
# -*- coding: utf-8 -*-
"""
OpenCV 실전 데모 (업그레이드)
기존: ① 이미지 출력/ROI/도형 ② 카메라/동영상 루프 ③ 얼굴+캐릭터 오버레이
      ④ 반자동 문서 스캐너(사각 외곽→투시보정) ⑤ 카드 다중 검출→저장
추가: ⑥ Canny 트랙바 튜너 ⑦ ROI 마우스 선택/저장 ⑧ 눈 기반 기울기 보정 오버레이
"""

import os
import cv2
import numpy as np
import argparse
import math

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
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# -----------------------------
# ① 이미지 출력/ROI/도형
# -----------------------------
def demo_image_draw(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"이미지 불러오기 실패: {img_path}")
    h, w = img.shape[:2]
    # ROI 채우기
    y1,y2 = int(h*0.15), int(h*0.35)
    x1,x2 = int(w*0.15), int(w*0.40)
    img[y1:y2, x1:x2] = (0,255,255)  # BGR
    # 도형/텍스트
    cv2.line(img,(20,20),(w-20,20),(255,255,255),2,cv2.LINE_AA)
    cv2.rectangle(img,(int(w*0.60),int(h*0.15)),(int(w*0.90),int(h*0.35)),(0,0,255),3)
    cv2.rectangle(img,(int(w*0.60),int(h*0.40)),(int(w*0.90),int(h*0.60)),(0,128,0),-1)
    cv2.circle(img,(int(w*0.20),int(h*0.70)),60,(255,0,0),3)
    pts = np.array([[int(w*0.10),int(h*0.85)],[int(w*0.25),int(h*0.75)],[int(w*0.35),int(h*0.90)]],np.int32)
    cv2.polylines(img,[pts],True,(255,0,255),2,cv2.LINE_AA)
    cv2.fillPoly(img,[pts + np.array([0,-50])],(128,0,128))
    cv2.putText(img,"OpenCV Demo",(20,h-20),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
    cv2.imshow("Image Draw Demo", img); cv2.waitKey(0); cv2.destroyAllWindows()

# -----------------------------
# ② 카메라/동영상 루프
# -----------------------------
def demo_video_loop(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"VideoCapture 열기 실패: {source}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("더 이상 가져올 프레임이 없습니다."); break
        cv2.imshow("Video Loop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# -----------------------------
# ③ 얼굴 검출 + 캐릭터 PNG 오버레이 (기본)
# -----------------------------
def overlay_transparent(bg, overlay_rgba, x, y, w, h):
    """bg(BGR)에 overlay_rgba(RGBA)를 (x,y)에 w*h로 알파블렌딩"""
    ov = cv2.resize(overlay_rgba,(w,h),interpolation=cv2.INTER_AREA)
    if ov.shape[2]==3:
        alpha = np.ones((h,w),dtype=float); ov_bgr = ov
    else:
        alpha = ov[:,:,3]/255.0; ov_bgr = ov[:,:,:3]
    y1,y2 = max(0,y), min(bg.shape[0], y+h)
    x1,x2 = max(0,x), min(bg.shape[1], x+w)
    ov_y1,ov_y2 = y1-y, y1-y+(y2-y1)
    ov_x1,ov_x2 = x1-x, x1-x+(x2-x1)
    if y1>=y2 or x1>=x2: return bg
    roi = bg[y1:y2, x1:x2]
    a = alpha[ov_y1:ov_y2, ov_x1:ov_x2][:,:,None]
    blended = (a*ov_bgr[ov_y1:ov_y2, ov_x1:ov_x2] + (1-a)*roi).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended
    return bg

def face_overlay_video(source, overlay_png, scale=1.2):
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml"))
    if face_cascade.empty(): raise RuntimeError("HaarCascade 로드 실패")
    ov = cv2.imread(overlay_png, cv2.IMREAD_UNCHANGED)
    if ov is None: raise FileNotFoundError(f"오버레이 PNG 불러오기 실패: {overlay_png}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"VideoCapture 열기 실패: {source}")
    while True:
        ret, frame = cap.read()
        if not ret: print("더 이상 가져올 프레임이 없습니다."); break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
        for (x,y,w,h) in faces:
            W,H = int(w*scale), int(h*scale)
            X = x - (W - w)//2
            Y = y - int(H*0.35)
            frame = overlay_transparent(frame, ov, X, Y, W, H)
        cv2.imshow("Face + Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# -----------------------------
# ④ 반자동 문서 스캐너
# -----------------------------
def scan_document(image_path, out_path=None, show_steps=False):
    orig = cv2.imread(image_path)
    if orig is None: raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")
    ratio = 800.0 / orig.shape[1]
    img = cv2.resize(orig, (800, int(orig.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray,60,180); edges = cv2.dilate(edges,np.ones((3,3),np.uint8),1)
    contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4: doc_quad = approx.reshape(4,2); break
    if doc_quad is None: raise RuntimeError("사각 외곽을 찾지 못했습니다.")
    doc_quad = (doc_quad/ratio).astype(np.float32)
    warped = four_point_transform(orig, doc_quad)
    scanned = cv2.adaptiveThreshold(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY),
                                    255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)
    if out_path: cv2.imwrite(out_path, scanned)
    if show_steps:
        cv2.imshow("edges", edges); cv2.imshow("warped", warped); cv2.imshow("scanned", scanned)
        cv2.waitKey(0); cv2.destroyAllWindows()
    return scanned

# -----------------------------
# ⑤ 카드 다중 검출 → 개별 저장
# -----------------------------
def split_cards(image_path, out_dir="cards_out", min_area=8000):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")
    draw = img.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(5,5),0)
    edges = cv2.Canny(gray,60,180); edges = cv2.dilate(edges,np.ones((3,3),np.uint8),1)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx=1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)!=4: continue
        quad = approx.reshape(4,2).astype(np.float32)
        warped = four_point_transform(img, quad)
        card_norm = cv2.resize(warped,(480,300),interpolation=cv2.INTER_AREA)
        save_path = os.path.join(out_dir, f"card_{idx:02d}.png")
        cv2.imwrite(save_path, card_norm); idx+=1
        cv2.polylines(draw,[approx],True,(0,255,0),2)
    cv2.imshow("Detected Cards", draw); cv2.waitKey(0); cv2.destroyAllWindows()
    print(f"총 {idx-1}개 저장됨 → {os.path.abspath(out_dir)}")

# -----------------------------
# (신규) ⑥ Canny 트랙바 튜너
# -----------------------------
def canny_tuner(image_path):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("CannyTuner")
    cv2.createTrackbar("low","CannyTuner",60,255,lambda x: None)
    cv2.createTrackbar("high","CannyTuner",180,255,lambda x: None)
    while True:
        low = cv2.getTrackbarPos("low","CannyTuner")
        high = cv2.getTrackbarPos("high","CannyTuner")
        edge = cv2.Canny(gray, max(0,low), max(low+1,high))
        show = cv2.hconcat([img, cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("CannyTuner", show)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q') or key==27: break
    cv2.destroyAllWindows()

# -----------------------------
# (신규) ⑦ ROI 마우스 선택 → 저장
# -----------------------------
def roi_picker(image_path, out_dir="roi_out"):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"이미지 불러오기 실패: {image_path}")
    # 다중 ROI 선택 (드래그 후 Enter/Space로 확정, c로 취소, ESC 종료)
    rois = cv2.selectROIs("ROI Picker", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("ROI Picker")
    count=0
    for i,(x,y,w,h) in enumerate(rois):
        if w<=0 or h<=0: continue
        crop = img[y:y+h, x:x+w]
        save_path = os.path.join(out_dir, f"roi_{i+1:02d}.png")
        cv2.imwrite(save_path, crop); count+=1
    print(f"저장된 ROI: {count}개 → {os.path.abspath(out_dir)}")

# -----------------------------
# (신규) ⑧ 눈 검출 기반 각도 보정 오버레이
#      - 눈 두 개 중심을 이용해 기울기 추정 → 캐릭터 PNG 회전 후 합성
# -----------------------------
def rotate_rgba(img_rgba, angle_deg, scale=1.0):
    (h,w) = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int((h*sin)+(w*cos)), int((h*cos)+(w*sin))
    M[0,2] += (nW/2) - w/2
    M[1,2] += (nH/2) - h/2
    rot = cv2.warpAffine(img_rgba, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    return rot

def face_overlay_video_rotated(source, overlay_png, scale=1.3):
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml"))
    eye_cascade  = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_eye_tree_eyeglasses.xml"))
    if face_cascade.empty() or eye_cascade.empty(): raise RuntimeError("Cascade 로드 실패")
    ov_rgba = cv2.imread(overlay_png, cv2.IMREAD_UNCHANGED)
    if ov_rgba is None: raise FileNotFoundError(f"오버레이 PNG 불러오기 실패: {overlay_png}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"VideoCapture 열기 실패: {source}")
    while True:
        ret, frame = cap.read()
        if not ret: print("더 이상 가져올 프레임이 없습니다."); break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
        for (x,y,w,h) in faces:
            # 눈 검출(얼굴 ROI 기준)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5, minSize=(20,20))
            angle_deg = 0.0
            if len(eyes) >= 2:
                # 좌우 두 눈 가운데 두 개 선택(가로 좌표로 정렬)
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                (ex1,ey1,ew1,eh1) = eyes[0]
                (ex2,ey2,ew2,eh2) = eyes[1]
                c1 = (x + ex1 + ew1//2, y + ey1 + eh1//2)
                c2 = (x + ex2 + ew2//2, y + ey2 + eh2//2)
                dy = c2[1]-c1[1]; dx = c2[0]-c1[0]
                angle_deg = math.degrees(math.atan2(dy, dx))
            # 얼굴 박스 기준으로 확대 배치
            W,H = int(w*scale), int(h*scale)
            X = x - (W - w)//2
            Y = y - int(H*0.35)
            # 캐릭터를 각도에 맞춰 회전
            ov_rot = rotate_rgba(ov_rgba, angle_deg)
            frame = overlay_transparent(frame, ov_rot, X, Y, W, H)
        cv2.imshow("Face + Rotated Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 핵심 데모(플러스)")
    p.add_argument("--mode", required=True,
                   choices=["image_draw","video","face","scan","cards","canny","roi","face_rot"],
                   help="실행 모드")
    p.add_argument("--image", help="이미지 경로 (image_draw/scan/cards/canny/roi)")
    p.add_argument("--source", default="0", help="0 또는 동영상 파일 경로 (video/face/face_rot)")
    p.add_argument("--overlay", help="캐릭터 PNG 경로 (face/face_rot)")
    p.add_argument("--out", help="scan 결과 저장 경로")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode=="image_draw":
        if not args.image: raise ValueError("--image 필요")
        demo_image_draw(args.image)

    elif args.mode=="video":
        src = 0 if args.source=="0" else args.source
        demo_video_loop(src)

    elif args.mode=="face":
        if not args.overlay: raise ValueError("--overlay PNG 필요")
        src = 0 if args.source=="0" else args.source
        face_overlay_video(src, args.overlay, scale=1.2)

    elif args.mode=="scan":
        if not args.image: raise ValueError("--image 필요")
        scan_document(args.image, out_path=args.out or "scanned.png", show_steps=True)

    elif args.mode=="cards":
        if not args.image: raise ValueError("--image 필요")
        split_cards(args.image, out_dir="cards_out")

    elif args.mode=="canny":
        if not args.image: raise ValueError("--image 필요")
        canny_tuner(args.image)

    elif args.mode=="roi":
        if not args.image: raise ValueError("--image 필요")
        roi_picker(args.image, out_dir="roi_out")

    elif args.mode=="face_rot":
        if not args.overlay: raise ValueError("--overlay PNG 필요")
        src = 0 if args.source=="0" else args.source
        face_overlay_video_rotated(src, args.overlay, scale=1.3)

if __name__=="__main__":
    main()

