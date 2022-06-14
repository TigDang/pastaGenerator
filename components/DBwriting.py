def pullPastas(connect, titles):
    with connect.cursor() as cursor:
        for title in titles:
            sql = "INSERT INTO pasta (pasta_name) VALUES ('{}')".format(title)
            cursor.execute(sql)

def pullIngridients(connect, distinctIngridients):
    with connect.cursor() as cursor:
        for ingr in distinctIngridients:
            sql = "INSERT INTO ingredient (ingr_name) VALUES ('{}')".format(ingr)
            cursor.execute(sql)

def pullRecipies(connect, titles, ingredients):
    pastaID = 0
    ingrID = 0
    pastaIDquery = "SELECT pasta_id FROM pasta WHERE pasta_name = '{}'"
    ingrIDquery = "SELECT ingr_id FROM ingredient WHERE ingr_name = '{}'"
    with connect.cursor() as cursor:
        for i in range(len(titles)):
            for ingr in ingredients[i]:
                cursor.execute(pastaIDquery.format(titles[i]))
                pastaID = cursor.fetchall()[0][0]
                cursor.execute(ingrIDquery.format(ingr))
                ingrID = cursor.fetchall()[0][0]

                print(titles[i], ingr, pastaID, ingrID, sep='~')

                sql = "INSERT INTO recipe (pasta_assoc_id, ingrd_assoc_id) VALUES ('{}', '{}')".format(pastaID,
                                                                                                       ingrID)
                cursor.execute(sql)
