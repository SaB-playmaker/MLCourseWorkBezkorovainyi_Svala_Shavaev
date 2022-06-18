from bs4 import BeautifulSoup
soup = BeautifulSoup("https://rozetka.com.ua/ua/notebooks/c80004/#search_text=%D0%BD%D0%BE%D1%83%D1%82%D0%B1%D1%83%D0%BA ", 'html.parser')
	
print(soup.prettify())
soup.find_all('a')
for link in soup.find_all('a'):
    print(link.get('href'))
for element in HTML_data:
    sub_data = []
    for sub_element in element:
        try:
            sub_data.append(sub_element.get_text())
        except:
            continue
    data.append(sub_data)
  
# Storing the data into Pandas
# DataFrame 
dataFrame = pd.DataFrame(data = data, columns = list_header)
   
# Converting Pandas DataFrame
# into CSV file
dataFrame.to_csv('laptop.csv')

