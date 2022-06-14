import numpy as np
import components.gettingPictureText as picText
import components.mysqlConnector as mysql
import components.DBwriting as dbw

# Open
c = mysql.Connect()

# dbw.pullPastas(c, picText.titles)

# Making array of distinct ingredients

distIngr = []
for arr in picText.ingridients:
    for i in arr:
        distIngr.append(i)

distIngr = set(distIngr)

dbw.pullIngridients(c, distIngr)

print(distIngr)


queryText = "SELECT * FROM ingredient"
with c.cursor() as cursor:
    cursor.execute(queryText)
    result = cursor.fetchall()
    for row in result:
           print(row)

# Commit inserting
c.commit()

# And close
mysql.Disconnect(c)


if __name__ == '__main__':

    print("Наименования: " + str(picText.titles), "Количество наименований: " + str(len(picText.titles)),
    sep= '\n')

    print("Ингридиенты: " + str(picText.ingridients), "Количество ингридиентов: " + str(len(picText.ingridients)),
    sep='\n')

