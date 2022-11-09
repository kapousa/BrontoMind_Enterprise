#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
from __future__ import print_function
from mailmerge import MailMerge
from datetime import date

# Define the templates - assumes they are in the same directory as the code
template_1 = "Slonos_Labs_BrontoMind_APIs_document_template.docx"
template_2 = "Slonos_Labs_BrontoMind_APIs_document_template.docx"

# Show a simple example
document_1 = MailMerge(template_1)
print("Fields included in {}: {}".format(template_1,
                                         document_1.get_merge_fields()))

# Merge in the values
for i in range(0,2):
    document_1.merge(
        method_name='test',
        vesrion='V1',
        desc='V1',
        sample_reponse='V1',
        smaple_request='V1',
        url='V1')

# Save the document as example 1
document_1.write('example1.docx')

# Try example number two where we create multiple pages
# Define a dictionary for 3 customers
cust_1 = {
    'method_name':'test',
    'vesrion':'V1',
    'desc':'V1',
    'sample_reponse':'V1',
    'smaple_request':'V1',
    'url':'V1'
}

cust_2 = {
    'method_name':'test',
    'vesrion':'V1',
    'desc':'V1',
    'sample_reponse':'V1',
    'smaple_request':'V1',
    'url':'V1'
}

cust_3 = {
    'method_name':'test',
    'vesrion':'V1',
    'desc':'V1',
    'sample_reponse':'V1',
    'smaple_request':'V1',
    'url':'V1'
}

document_2 = MailMerge(template_1)
document_2.merge_templates([cust_1, cust_2, cust_3], separator='textWrapping_break')
document_2.write('example2.docx')


# repeate section
values = {'method_name':'test',
    'vesrion':'V1',
    'desc':'V1',
    'sample_reponse':'V1',
    'smaple_request':'V1',
    'url':'V1'}

with MailMerge(template_1) as word_doc:
    for merge_field in word_doc.get_merge_fields():
       word_doc.merge(merge_field = values[merge_field])
    word_doc.write("exsample4.docx")


# Final Example includes a table with the sales history

sales_history = [{
    'vesrion': 'Red Shoes',
    'desc': '$10.00',
    'sample_reponse': '2500',
    'smaple_request': '$25,000.00'
}, {
    'vesrion': 'Green Shirt',
    'desc': '$20.00',
    'sample_reponse': '10000',
    'smaple_request': '$200,000.00'
}, {
    'vesrion': 'Purple belt',
    'desc': '$5.00',
    'sample_reponse': '5000',
    'smaple_request': '$25,000.00'
}]

document_3 = MailMerge(template_2)
document_3.merge(**cust_2)
document_3.merge_rows('vesrion', sales_history)
document_3.write('example3.docx')
