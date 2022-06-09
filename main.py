import components.gettingPictureText as picText


if __name__ == '__main__':

    print("Наименования: " + str(picText.titles), "Количество наименований: " + str(len(picText.titles)),
    sep= '\n')

    print("Ингридиенты: " + str(picText.ingridients), "Количество ингридиентов: " + str(len(picText.ingridients)),
    sep='\n')