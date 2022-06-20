import mysqlConnector as mysql
import numpy as np


def GetInterpretArray(c, arr):
    result = []
    getIngrNameQuery = "SELECT ingr_name FROM ingredient WHERE ingr_id = {}"
    with c.cursor() as cursor:
        for i in range(len(arr)):
            if arr[i] == 1:
                cursor.execute(getIngrNameQuery.format(i))
                result.append(cursor.fetchall()[0])
    return result

c = mysql.Connect()

# Queries
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

if __name__ == '__main__':
    print(dataset)
    print(GetInterpretArray(c, dataset[0]))
