from dataset import Dataset
import pandas as pd


file_path_save_data = 'data/processed/' 
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)




## for 100k
       
path = '~/.surprise_data/ml-100k/ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
list_of_cats = {}

df1 = df[['id','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
for row in df.itertuples(index=True, name='Pandas'):
    id = str(getattr(row, "id"))
    cate_x = [getattr(row, "cat1"),getattr(row, "cat2"),getattr(row, "cat3"),getattr(row, "cat4"),getattr(row, "cat5"),getattr(row, "cat6"),getattr(row, "cat7"),getattr(row, "cat8"),getattr(row, "cat9"),getattr(row, "cat10"),getattr(row, "cat11"),getattr(row, "cat12"),getattr(row, "cat13"),getattr(row, "cat14"),getattr(row, "cat15"),getattr(row, "cat16"),getattr(row, "cat17"),getattr(row, "cat18"),getattr(row, "cat19"),]
    list_of_cats[id] = cate_x

## for small-100k

# def convert(listx):
#     #listx : ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']
#     list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if '(no genres listed)' in listx:
#         list[0] = 1
#     if 'Action' in listx:
#         list[1] = 1
#     if 'Adventure' in listx:
#         list[2] = 1
#     if 'Animation' in listx:
#         list[3] = 1
#     if 'Children' in listx:
#         list[4] = 1
#     if 'Comedy' in listx:
#         list[5] = 1
#     if 'Crime' in listx:
#         list[6] = 1
#     if 'Documentary' in listx:
#         list[7] = 1
#     if 'Drama' in listx:
#         list[8] = 1
#     if 'Fantasy' in listx:
#         list[9] = 1
#     if 'Film-Noir' in listx:
#         list[10] = 1
#     if 'Horror' in listx:
#         list[11] = 1
#     if 'Musical' in listx:
#         list[12] = 1
#     if 'Mystery' in listx:
#         list[13] = 1
#     if 'Romance' in listx:
#         list[14] = 1
#     if 'Sci-Fi' in listx:
#         list[15] = 1
#     if 'Thriller' in listx:
#         list[16] = 1
#     if 'War' in listx:
#         list[17] = 1
#     if 'Western' in listx:
#         list[18] = 1
#     return list
#
#
# path = '~/.surprise_data/ml-latest-small/movies.csv'
# df = pd.read_csv(path, sep=",", encoding="iso-8859-1", names=['id','name','CATE'])
# df = df[['id','CATE']]
# list_of_cats = {}
#
# for row in df.itertuples(index=True, name='Pandas'):
#
#     id = str(getattr(row, "id"))
#
#     cate = getattr(row, "CATE")
#
#     # change cate(str) into the list of cate(list)
#     cate = cate.split('|')
#
#     # set a map function from cate(word) to cate(0or1)
#     cate = convert(cate)
#
#     list_of_cats[id] = cate

## for 1m



# print(df1)
# df1.to_csv(file_path_save_data+'itemswithcat.csv', sep='\t', encoding='utf-8')
# print(list_of_cats)