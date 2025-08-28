실행 예

이미지·도형:
python main_opencv_demo.py --mode image_draw --image sample.jpg

카메라 루프:
python main_opencv_demo.py --mode video --source 0

얼굴 + 캐릭터 오버레이(투명 PNG 필요):
python main_opencv_demo.py --mode face --source 0 --overlay character.png

문서 스캔(사각 프레임이 있는 문서 사진):
python main_opencv_demo.py --mode scan --image doc.jpg --out scanned.png

카드/명함 여러 장 자동 분할:
python main_opencv_demo.py --mode cards --image cards.jpg

사용 팁

Haar 분류기는 cv2.data.haarcascades에서 자동 로드됩니다. 얼굴이 작게 나오면 카메라와 거리를 조절하거나 조명을 밝게 하십시오.

오버레이 PNG는 알파(투명) 채널이 있는 파일을 권장합니다(예: 귀·모자, 캐릭터 얼굴).

문서/카드 검출 실패 시: 해상도 높이기, 배경과의 명암 대비 키우기, 외곽이 프레임 안에 충분히 들어오도록 재촬영하십시오.

필요 시 detectMultiScale의 scaleFactor/minNeighbors/minSize를 조절하십시오

실행 예(복사해 바로 쓰시면 됩니다)

이미지·도형:
python main_opencv_demo_plus.py --mode image_draw --image sample.jpg

카메라 루프:
python main_opencv_demo_plus.py --mode video --source 0

얼굴+캐릭터(기본 오버레이):
python main_opencv_demo_plus.py --mode face --source 0 --overlay character.png

얼굴 기울기 보정 오버레이(신규):
python main_opencv_demo_plus.py --mode face_rot --source 0 --overlay character.png

문서 스캔:
python main_opencv_demo_plus.py --mode scan --image doc.jpg --out scanned.png

카드 자동 분할:
python main_opencv_demo_plus.py --mode cards --image cards.jpg

Canny 트랙바 튜너(신규):
python main_opencv_demo_plus.py --mode canny --image doc.jpg

ROI 마우스 선택/저장(신규):
python main_opencv_demo_plus.py --mode roi --image big_image.jpg

사용 팁

face_rot는 눈 두 개가 검출될 때 각도를 계산합니다. 조명이 어두우면 검출률이 떨어지므로 밝게 촬영해 주십시오.

캐릭터 PNG는 알파 채널이 있는 것을 권장합니다(투명 배경).

문서/카드 검출 실패 시: 해상도↑, 배경 대비↑, Canny 튜너로 적정 임계값을 먼저 찾은 뒤 적용하십시오


업그레이드 버전을 캔버스에 올려 두었습니다.
파일명: opencv_advanced_demo.py (미디어파이프 랜드마크 정밀 오버레이 + HSV/파이프라인 튜너 포함)

설치
pip install opencv-python numpy mediapipe

바로 실행 예시

랜드마크 정밀 오버레이(카메라 0번, PNG 필요):

python opencv_advanced_demo.py --mode face_landmark --source 0 --overlay character.png


HSV 마스크 튜너(설정 저장까지):

python opencv_advanced_demo.py --mode hsv_tuner --image cards.jpg --save hsv_params.json


전처리 파이프라인 튜너(문서/카드용):

python opencv_advanced_demo.py --mode pipeline_tuner --image doc.jpg


문서 스캔(투시보정):

python opencv_advanced_demo.py --mode scan --image doc.jpg --out scanned.png


카드 자동 분할·저장:

python opencv_advanced_demo.py --mode cards_auto --image cards.jpg

사용 팁

face_landmark는 눈 좌표로 각도와 스케일을 자동 보정합니다. 조명 밝게, 얼굴 정면에서 효과가 좋습니다.

hsv_tuner 창에서 s 키로 파라미터를 JSON 저장합니다(마스크 자동화에 활용).

pipeline_tuner는 블러/캐니/모폴로지/적응 임계값을 실시간 미리보기로 튜닝해 최적값을 찾는 용도입니다
