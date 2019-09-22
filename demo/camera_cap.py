import cv2
cap = cv2.VideoCapture(0)

cascade_path = "haarcascade_frontalface_alt.xml"

while True:
    ret, im = cap.read()

    # ここからのコードを変えながら、微調整するとオリジナルになると思います。
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=(80, 80))  # minNeighborsは人数
    print(faces)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    import matplotlib.pyplot as plt

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    blur = cv2.GaussianBlur(im, (0, 0), 1)
    cv2.imshow('camera capture', blur)
    key = cv2.waitKey(10)
    # カメラはESCキーで終了できるように。
    if key == 27:
        break

    # 一旦画像削除の命令
cap.release()
# カメラが立ち上がっているので、全てのウィンドウを閉じる
cv2.destroyAllWindows()
