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

업그레이드 완료했습니다. 캔버스의 opencv_advanced_demo.py에 아래 신규 모드가 추가되어 있습니다.

새 기능 요약

face_landmark_multi : 여러 PNG(폴더/목록)로 다중 얼굴에 서로 다른 캐릭터 자동 배정.

hsv_apply : 저장한 HSV 파라미터(JSON) 를 이미지/폴더에 일괄 적용하여 *_masked.png/*_mask.png 생성.

split_cards_with_hsv 함수 : HSV 마스크로 전처리 후 카드 분할 정밀도 향상(필요 시 직접 호출).

scan_cam_auto : 웹캠에서 문서 사각형이 안정적으로 인식되면 자동 저장(간격/안정 프레임 수 조절).

cards_auto_dir : 폴더 내 이미지들을 일괄 카드 분할하여 이미지별 하위 폴더로 저장.

바로 실행 예시
# 1) 다중 얼굴-캐릭터
python opencv_advanced_demo.py --mode face_landmark_multi --source 0 --overlay_dir overlays/

# 2) HSV 적용(폴더 전체, JSON은 hsv_tuner에서 저장한 파일)
python opencv_advanced_demo.py --mode hsv_apply --indir samples/ --params hsv_params.json --outdir hsv_out/

# 3) HSV 적용(단일 파일)
python opencv_advanced_demo.py --mode hsv_apply --image cards.jpg --params hsv_params.json

# 4) 실시간 문서 스캔 자동 저장(2초 간격, 10프레임 안정 필요)
python opencv_advanced_demo.py --mode scan_cam_auto --source 0 --outdir scans --interval 2.0 --stable 10

# 5) 카드 분할(폴더 일괄)
python opencv_advanced_demo.py --mode cards_auto_dir --indir cards_batch/ --outdir cards_out/

팁

face_landmark_multi는 좌→우 얼굴 순서대로 PNG를 순환 배정합니다. 특정 매핑이 필요하시면 규칙을 알려주십시오.

scan_cam_auto에서 s 키로 수동 저장, q 키로 종료 가능합니다. 흔들림이 크면 --stable을 높이거나 --interval을 늘려 주세요.

HSV 파라미터 JSON을 카드 분할에 직접 쓰시려면 split_cards_with_hsv("cards.jpg", "hsv_params.json")처럼 함수 호출만 추가하시면 됩니다

오류 수정

요청하신 대로 오류 재현(SystemExit: 2) 방지를 위해 캔버스의 스크립트를 전면 수정했습니다. 이제 인자 없이 실행해도 --mode help로 자동 전환되어 사용법/예시를 출력하고 정상 종료합니다. 또한 --mode selftest를 추가하여 GUI/카메라 없이도 기본 기능을 점검할 수 있도록 단위 테스트를 포함했습니다.

수정 핵심

--mode 기본값을 help로 변경 → 필수 인자 누락으로 인한 강제 종료(코드 2) 방지.

인자 오류 시 친절한 가이드 + 실행 예시를 함께 출력.

selftest 모드 추가(창/카메라 사용 없음):

order_points 정렬 검증

회전/알파블렌딩 스모크 테스트

HSV 마스크 동작성 테스트

빠른 확인
# 도움말 자동 표출(정상 종료)
python opencv_advanced_demo.py

# 단위 테스트(창/카메라 없음, 통과 시 메시지 표시)
python opencv_advanced_demo.py --mode selftest
