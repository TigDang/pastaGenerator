import components.gettingPictureText as picText
import  components.mysqlConnector as mysql

# Open
c = mysql.Connect()
select_movies_query = "SELECT * FROM ingredient"
with c.cursor() as cursor:
    cursor.execute(select_movies_query)
    result = cursor.fetchall()
    for row in result:
           print(row)

# And close
mysql.Disconnect(c)


if __name__ == '__main__':

    print("Наименования: " + str(picText.titles), "Количество наименований: " + str(len(picText.titles)),
    sep= '\n')

    print("Ингридиенты: " + str(picText.ingridients), "Количество ингридиентов: " + str(len(picText.ingridients)),
    sep='\n')

