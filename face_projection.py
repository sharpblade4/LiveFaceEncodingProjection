from typing import Tuple, List

import face_recognition
import cv2
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from contextlib import contextmanager
import os
import matplotlib.pyplot as plt


@contextmanager
def webcam_stream():
    video_capture = cv2.VideoCapture(0)
    yield video_capture
    video_capture.release()


def projection(vectors: np.ndarray) -> np.ndarray:
    # return TSNE(n_components=2).fit_transform(vectors)
    return PCA(n_components=2).fit_transform(vectors)


def get_family_face_encodings(
    family_faces_path: str, force_new: bool = False
) -> Tuple[List[List[float]], List[str]]:
    if (
        not force_new
        and os.path.exists("encodings.npy")
        and os.path.exists("names.npy")
    ):
        return np.load("encodings.npy").tolist(), np.load("names.npy").tolist()
    encodings = []
    names = []
    for pic_name in os.listdir(family_faces_path):
        if pic_name.endswith((".png", ".jpg", ".jpeg")):
            member_name = ".".join(pic_name.split(".")[:-1])
            pic_path = os.path.join(family_faces_path, pic_name)
            image = face_recognition.load_image_file(pic_path)
            identified_faces = face_recognition.face_encodings(image)
            if identified_faces:
                encodings.append(face_recognition.face_encodings(image)[0])
                names.append(member_name)
            else:
                print(f"{member_name} face isn't appearing well in image.")
    return encodings, names


def project_to_2d(frame, family_encodings: List[List[float]]) -> Tuple[float, float]:
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    last_projection = projection(
        np.array(family_encodings + face_encodings, dtype=np.float128)
    )[-1]
    return last_projection


def live_map() -> None:
    plt.axis([0, 10, 0, 1])
    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        plt.pause(0.05)
    plt.show()


def research_encoding() -> None:
    known_face_encodings, known_face_names = get_family_face_encodings(
        family_faces_path="./job_faces", force_new=True
    )
    twins_encodings_names = [
        (e, known_face_names[i])
        for i, e in enumerate(known_face_encodings)
        if known_face_names[i].lower() in {"chaim", "shlomi", "zvi", "edna"}
    ]
    f, axs = plt.subplots(1, 4, sharey=True, sharex=True)
    for i, (enc, nam) in enumerate(twins_encodings_names):
        axs[i].text(0, 0, nam)
        axs[i].imshow(np.array(enc).reshape(16, 8))

    for i in range(len(known_face_encodings)):
        axs[i % 2][i % 13].text(0, 0, known_face_names[i])
        axs[i % 2][i % 13].imshow(np.array(known_face_encodings[i]).reshape(16, 8))
    plt.show()


def create_face_map(
    known_face_encodings: List[List[float]], known_face_names: List[str]
) -> None:
    dots = projection(np.array(known_face_encodings, dtype=np.float128))
    plt.scatter(dots[:, 0], dots[:, 1], c="blue")
    for i, (x, y) in enumerate(dots):
        plt.text(x, y, known_face_names[i])
    plt.axis("off")
    plt.show(block=False)


def process_live_feed(video_capture: cv2.VideoCapture) -> None:
    known_face_encodings, known_face_names = get_family_face_encodings(
        family_faces_path="./family_faces"
    )
    create_face_map(known_face_encodings, known_face_names)

    face_locations = []
    face_names = []
    process_this_frame = True
    pause = False
    d = None
    while True:
        _, frame = video_capture.read()

        if frame is None:
            print("Camera is not accessible. Exiting.")
            exit(-1)
        scale_factor = 4
        small_frame = cv2.resize(
            frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor
        )

        rgb_small_frame = small_frame[:, :, ::-1]  # OpenCV BGR --> face_recognition RGB

        if not pause:
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding
                    )
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    x, y = project_to_2d(rgb_small_frame, known_face_encodings)
                    if d is not None:
                        d.remove()
                    d = plt.scatter(x, y, c="red")
                    plt.draw()

            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= scale_factor
                right *= scale_factor
                bottom *= scale_factor
                left *= scale_factor
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Video", frame)

        key_hook = cv2.waitKey(1) & 0xFF
        if key_hook == ord("q"):
            break
        elif key_hook == ord("c"):
            pause = not pause


if __name__ == "__main__":
    research_encoding()
    with webcam_stream() as video:
        process_live_feed(video)
        cv2.destroyAllWindows()
