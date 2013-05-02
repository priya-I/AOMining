'''
Created on Mar 6, 2013

@author: priya
'''

'''S_ID User_ID Dest_ID Timestamp Action
last four digits in S_ID: MM_SS
'''
with open('inbound_3.log') as log:
    for row in log:
        row.split()[0]
        
 