import os, glob, cv2

cascade_path = "haarcascades/"
src_images_path = "images/src"
dict_images_path = "images/dist"

color = (255, 255, 255)
#color = (0, 0, 0)


def get_cascades_filenames() -> []:
    cascades = []
    for fpath in glob.glob(f'{cascade_path}/*'):
        cascades.append(fpath)
    return cascades

def face_detect_to_make_image(cascade_filepath:str) -> None:
    for fpath in glob.glob(f'{src_images_path}/*'):

        image = cv2.imread(fpath)

        # ここからのコードを変えながら、微調整するとオリジナルになると思います。
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_filepath)

        # 物体人正規
        #faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(80, 80))  # minNeighborsは人数
        faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        print(faces)

        src_filename = os.path.basename(fpath)
        cascade_filename = os.path.basename(cascade_filepath)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            dict_filename = f'processed_{src_filename}'
            cascade_type = cascade_filename.split("haarcascade_")[1].split(".xml")[0]

            dict_dirpath = f'{dict_images_path}/{cascade_type}'
            os.makedirs(dict_dirpath, exist_ok=True)

            dict_filepath = f'{dict_dirpath}/{dict_filename}'
            cv2.imwrite(dict_filepath, image)
        else:
            print(f"{fpath}:Not faces!")

def main():
    cascades = get_cascades_filenames()
    for cascade in cascades:
        face_detect_to_make_image(cascade)

if __name__ == "__main__":
    main()