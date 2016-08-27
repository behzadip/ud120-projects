#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


#count = 0
#for key in enron_data.keys():
#  count += int(enron_data[key]['poi'] == True)
#
#print count

#with open('../final_project/poi_names.txt') as f:
#    lines = f.readlines()
#print len(lines)

#print enron_data['PRENTICE JAMES']['total_stock_value']
#print enron_data['COLWELL WESLEY']['from_this_person_to_poi'] 
#print enron_data['SKILLING JEFFREY K']['exercised_stock_options']
#print enron_data['SKILLING JEFFREY K']['total_payments']
#print enron_data['FASTOW ANDREW S']['total_payments']
#print enron_data['LAY KENNETH L']['total_payments']


#count = 0
#count2 = 0
#for key in enron_data.keys():
#  if type(enron_data[key]['salary']) is int:
#      count += 1
#  if enron_data[key]['email_address'] != 'NaN':
#      count2 += 1
#print count
#print count2

count = 0
count2 = 0
for key in enron_data.keys():
  if enron_data[key]['poi'] == True:
    count += 1  
    if enron_data[key]['total_payments'] == 'NaN':
      count2 += 1
print float(count2) / float(count)
print count
print count2
