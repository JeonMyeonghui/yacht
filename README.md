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
