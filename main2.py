import classifier
import data_loader
from tkinter  import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment


def main():
    # data_loader.main()
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    sound = AudioSegment.from_mp3(filename)
    dst = filename.split(".mp3")[0] + ".wav"
    sound.export(dst, format="wav")
    model = classifier.classifier(dst)


if __name__ == '__main__':
    main()