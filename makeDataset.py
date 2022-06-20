import components.mysqlConnector as mysql
import numpy as np

c = mysql.Connect()

pastasIDquery = "SELECT DISTINCT pasta_assoc_id FROM recipe"
ingrsIDquery = "SELECT MAX(ingrd_assoc_id) FROM recipe"
recipesQuery = "SELECT ingrd_assoc_id FROM recipe WHERE pasta_assoc_id = {}"
with c.cursor() as cursor:

    # Learn count of pastas
    cursor.execute(pastasIDquery)
    pastasIDs = cursor.fetchall()
    countOfPastas = len(pastasIDs)

    # Learn count of ingrs
    cursor.execute(ingrsIDquery)
    countOfIngrs = cursor.fetchall()[0][0] + 1

    # Make numpy array of dataset
    dataset = np.zeros((countOfPastas, countOfIngrs))

    # Coursor at the row of pasta
    i = 0

    # Fill the dataset's rows the data, pull the ones to the position on ingrs
    for pastaID in pastasIDs:
        cursor.execute(recipesQuery.format(pastaID[0]))
        ingrsOfThatPasta = cursor.fetchall()
        for ingrID in ingrsOfThatPasta:
            dataset[i][ingrID] = 1
        i += 1

np.set_printoptions(precision=0)
print(dataset)