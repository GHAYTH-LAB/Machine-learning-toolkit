import graphviz  
from sklearn.tree import export_graphviz  
dot_data= export_graphviz(model.estimators_[10], out_file = None, 
                    feature_names = X.columns,  
                      class_names = ['0', '1'],  
                      filled = True, rounded = True,  
                      special_characters = True, impurity = True)  
graph = graphviz.Source(dot_data, format='png')  
graph  