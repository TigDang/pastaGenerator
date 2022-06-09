import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

if __name__ == '__main__':
    img = cv2.imread('../src/menuImageJoined.png')
else:
    img = cv2.imread('src/menuImageJoined.png')


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
config = r'--oem 3--psm 6'

# Getting raw text from the picture, uses
rawPictureText = pytesseract.image_to_string(img, config=config, lang='rus')

# Getting titles from raw text (with slash at the end)
titlesRaw = re.findall(r'[А-Я ]+\/', rawPictureText)

# Clear strings from slashes TODO: simplify
titles = []
for w in titlesRaw:
    titles.append(w.replace('/', ''))

# Getting ingridients, located in the cases
ingridientsRaw = re.findall(r"^\(([^)]+)\)", rawPictureText, flags=re.MULTILINE)

ingridients = []
for i in ingridientsRaw:
    ingridients.append(re.split(r", ", i.replace("\n", "").lower(), flags=re.MULTILINE))

fulldata = []
for i in range(len(titles)):
    fullstr = "\n" + titles[i] + ": \n"
    fullstr += ", ".join(ingridients[i])
    fulldata.append(fullstr)


def ShowRawText():
    print("Считанный текст: \n" + rawPictureText + "\n")


def ShowParsedText():
    print("Обработанный текст:")
    for raw in fulldata:
        print(raw)


# If that file is running
if __name__ == '__main__':
    # Output
    # print("Наименования: " + str(titles), "Количество наименований: " + str(len(titles)),
    # sep= '\n')
    #
    # print("Ингридиенты: " + str(ingridients), "Количество ингридиентов: " + str(len(ingridients)),
    # sep='\n')
    ShowRawText()
    ShowParsedText()
