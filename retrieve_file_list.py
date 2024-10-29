import requests

parameters = {'shortname':'nasingh'}

url = 'http://cs7ns1.scss.tcd.ie'

try:
    response = requests.get(url,params=parameters)
    if response.status_code == 200:
        with open('file_list.txt', 'w') as f:
            f.write(response.text)
            print("Done")
    else:
        print("ERROR")
except Exception as e:
    print(f"Error : {e}")
