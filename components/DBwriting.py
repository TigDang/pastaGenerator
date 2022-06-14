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
    print('something')
    # TODO: make this