# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:52:07 2022

@author: Cheever
"""

# Testing Amazon Product Review Scraping

#Lets try with this Beautiful Soup tutorial

#Code source:https://www.geeksforgeeks.org/web-scraping-amazon-customer-reviews/?ref=rp

# import packages

import requests
from bs4 import BeautifulSoup

HEADERS = ({'User-Agent':
			'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
			AppleWebKit/537.36 (KHTML, like Gecko) \
			Chrome/90.0.4430.212 Safari/537.36',
			'Accept-Language': 'en-US, en;q=0.5'})

# user define function
# Scrape the data
def getdata(url):
	r = requests.get(url, headers=HEADERS)
	return r.text


def html_code(url):

	# pass the url
	# into getdata function
	htmldata = getdata(url)
	soup = BeautifulSoup(htmldata, 'html.parser')

	# display html code
	return (soup)


url = "https://www.amazon.com/Milk-Bone-Marosnacks-Treats-Sizes-40-Ounce/dp/B003PMQMK2/ref=cm_cr_arp_d_product_top?ie=UTF8&th=1"

soup = html_code(url)
print(soup)

# Customer Names


def cus_data(soup):
	# find the Html tag
	# with find()
	# and convert into string
	data_str = ""
	cus_list = []

	for item in soup.find_all("span", class_="a-profile-name"):
		data_str = data_str + item.get_text()
		cus_list.append(data_str)
		data_str = ""
	return cus_list


cus_res = cus_data(soup)
print(cus_res)

# Review Text

def cus_rev(soup):
	# find the Html tag
	# with find()
	# and convert into string
	data_str = " "

	for item in soup.find_all("div", class_="a-row a-spacing-small review-data"):
		data_str = data_str + item.get_text()

	result = data_str.split("\n")
	return (result)


rev_data = cus_rev(soup)
rev_result = []
for i in rev_data:
	if i is " ":
		pass
	else:
		rev_result.append(i)
rev_result
print(rev_result)

#Product Meta Data

def product_info(soup):

	# find the Html tag
	# with find()
	# and convert into string
	data_str = ""
	pro_info = []

	for item in soup.find_all("ul", class_="a-unordered-list a-nostyle a-vertical a-spacing-none detail-bullet-list"):
		data_str = data_str + item.get_text()
		pro_info.append(data_str.split("\n"))
		data_str = ""
	return pro_info


pro_result = product_info(soup)

# Filter the required data
for item in pro_result:
	for j in item:
		if j is "":
			pass
		else:
			print(j)



# Images

def rev_img(soup):

	# find the Html tag
	# with find()
	# and convert into string
	data_str = " "
	cus_list = []
	images = []
	for img in soup.findAll('img', class_="cr-lightbox-image-thumbnail"):
		images.append(img.get('src'))
	return images


img_result = rev_img(soup)
img_result

def stars(soup):
    
    data_str = " "
    star = []
    for img in soup.findAll('i', datahook="review-star-rating"):
        star.append(img.get('src'))
    return star

stars_result = stars(soup)
print(stars_result)