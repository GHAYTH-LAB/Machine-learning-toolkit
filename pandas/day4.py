"""import pandas as pd
Dict={
    "Game":["Forza Horizon","Fifa 19","red Dead redemption"]
    ,"Edition":[2019,2020,2022]
    ,"version":[3,4,8]
}
Dict=pd.DataFrame(Dict)
Dict.index=["info_1","info_2","info_3"]
print(Dict)
print("----------------")
print(Dict.loc["info_1"])
print(Dict.iloc[2])
print("-----------------")
print(Dict["Edition"])
print(Dict["Game"])
Dict.loc["info_1"]={
    "Game":"Pes"
    ,"Edition":2025
    ,"version":5   
}
print("-----------------")
print(Dict)
Dict["version"]=[21,17,2]
print("------------------")
print(Dict)
New_rows={
    "Game":["Blur","Gta San andreas"]
    ,"Edition":[2012,2006]
    ,"version":[14,20]
}
New_rows=pd.DataFrame(New_rows,index=["info_4","info_5"])
Dict=pd.concat([Dict,New_rows])
print(Dict)
print("---------------------")
Dict["Name"]=["Ghayth","Anass","Mohamed","Aziz","Rayen"]
print(Dict)
"""
import pandas as pd
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\google_playstore_2026.csv")
print(df)
print("---------------------")
print(df.iloc[0])
print("---------------------")
print(df.loc[1])
print(df.loc[[1,3,5]])
print("---------------------")
print(df.loc[2:5])
print("----------------------")
print(df[["App","Rating"]])
print("----------------------")
print(df.loc[0:4,["App","Rating","Category"]])
print("----------------------")
print(df.loc[120:125,["App","Category","Rating"]])