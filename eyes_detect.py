import cv2

def preprocessing():

    # 분석하기 위한 이미지 불러오기
    image = cv2.imread("image/jsw.jpg", cv2.IMREAD_COLOR)

    # 이미지가 존재하지 않으면, 에러 반환
    if image is None: return None, None

    # 이미지 크기 사이즈 변경
    image = cv2.resize(image, (700, 700))

    # 흑백사진으로 변경
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray1", gray)

    # 변환한 흑백사진으로부터 히스토그램 평활화
    gray = cv2.equalizeHist(gray)

    cv2.imshow("gray2", gray)

    return image, gray


# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 학습된 눈 검출기 사용하기
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

# 인식률을 높이기 위한 전처리 함수 호출
image, gray = preprocessing() # 전처리

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터 조절)
# 얼굴 검출은 히스토그램 평활화한 이미지 사용
# scaleFactor : 1.5
# minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 얼굴 검출 성공했다면
if faces.any():

    # 얼굴 위치값 가져오기
    x, y, w, h = faces[0]

    # 원본 이미지로부터 얼굴 영역 가져오기
    faces_image = image[y : y + h, x : x + w]

    # 눈 검출 수행하기(정확도 높이는 방법의 아래 파라미터를 조절함)
    # 눈 검출은얼굴 이미지 영역만 불러와 분석 수행
    # scaleFactor : 1.15
    # minNeighbors : 인근 유사 픽셀 발견 비율이 7 번 이상
    # flags : 0 => 더이상 사용하지 않는 인자값
    # 분석할 이미지의 최소 크기 : 가로 25, 세로 20
    eyes = eye_cascade.detectMultiScale(faces_image, 1.15, 7, 0, (25, 20))

    # 눈을 찾을 수 없다면
    if len(eyes) == 2:
        for ex, ey, ew, eh in eyes:
            # 눈 가운데 위치 값 가져오기
            center = (x + ex + ew // 2, y + ey + eh // 2)

            # 눈 중심에 원 그리기
            cv2.circle(image, center, 10, (0, 255, 0), 2)

    else:
        print("눈 미검출")

    # 얼굴 검출 사각형 그리기
    cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    # 사이즈 변경된 이미지로 출력하기
    cv2.imshow("MyFace", image)

# 입력받는 것 대기하기, 작성안하면 결과창이 바로 닫힘
cv2.waitKey(0)