import re
from django.utils.timezone import datetime
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import requests  # For REST Posts
import json  # For JSON conversion
import base64  # Convert Files for posting
import mimetypes  # Get the content Type
import ntpath  # Get the File Path
import pandas as pd
from pathlib import Path 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

baseURL = 'https://capture.ot2.opentext.com/cp-rest/session'
#EUbaseURL = 'https://capture.ot2.opentext.eu/cp-rest/session'
authUrl = "https://otdsauth.ot2.opentext.com/oauth2/token"

def auth():
    

    # Now create the Login request
    loginRequest = {}
    loginRequest['grant_type'] = 'client_credentials'
    loginRequest['username'] = 'ghutchin@opentext.com'
    loginRequest['password'] = 'Opentext123456!'
    loginRequest['sitename'] = 'OTPreSales2_0'
    loginRequest['subscriptionName'] = 'ghutchin'

    # Take the client secret from the developer console and convert it to base 64
    client = '25b9ba98-c8fc-4373-8256-dc72e55a4fa0'
    secret = 'f9b619e66df0492aa7d8a78afe51a799'
    clientSecret = client + ':' + secret
    csEncoded = base64.b64encode(clientSecret.encode())

    # You now need to decode the Base64 to a string version
    csString = csEncoded.decode('utf-8')

    # Add the Client Secret and content Type to the request header
    loginHeaders = {}
    loginHeaders['Content-Type'] = 'application/x-www-form-urlencoded'
    loginHeaders['Authorization'] = "Basic " + csString

    # Now post the request
    r = requests.post(authUrl, data=loginRequest, headers=loginHeaders)
    loginResponse = json.loads(r.text)

    accessToken = loginResponse['access_token']
    return accessToken

def upload_image(filename,accessToken):
    # Create the service headers
    serviceHeaders = {}
    serviceHeaders['Authorization'] = 'Bearer ' + accessToken
    serviceHeaders['Content-Type'] = 'application/hal+json; charset=utf-8'

    # Now we're going to upload the Image
    
    # Create Image Upload object

    imageUpload = {}

    # Open the file and convert it to a Base 64 String
    fileName = filename
    # we'll use this for later
    originalFileName = ntpath.basename(fileName)

    imageFile = open(fileName, 'rb').read()
    imageB64 = base64.encodebytes(imageFile)

    # Assign this string to a data element called data
    imageUpload['data'] = imageB64.decode('utf-8')

    # Now get the Mimetype of the file
    # add 0 to get ('image/tiff', None)
    mime = mimetypes.guess_type(fileName)[0]
    imageUpload['contentType'] = mime
    # Now convert the object to json
    uploadJson = json.dumps(imageUpload)

    # Now upload the image
    # base url for all commands

    
    # Now for the file resource
    uploadURL = baseURL + '/files'


    uploadRequest = requests.post(uploadURL, data=uploadJson, headers=serviceHeaders)
    
    uploadResponse = json.loads(uploadRequest.text)
    
    uploadFileID = uploadResponse['id']
    uploadContentType = uploadResponse['contentType']
    return {'uploadFileID': uploadFileID, 'uploadContentType' : uploadContentType}

## Convert the documents to single page images
def image_convert(FileID,ContentType,accessToken):
    serviceHeaders = {}
    serviceHeaders['Authorization'] = 'Bearer ' + accessToken
    serviceHeaders['Content-Type'] = 'application/hal+json; charset=utf-8'
    splitImages = []
    convertURL = baseURL + '/services/convertimages'
    
    convertImage = {}
    # First create the Service Props
    convertServiceProps = []
    # We can also use this for the Classify & Extract request
    enviroment = {}
    enviroment['name'] = 'Env'
    enviroment['value'] = "S"
    convertServiceProps.append(enviroment)

    convertImageProfile = {}
    convertImageProfile['name'] = 'Profile'
    convertImageProfile['value'] = 'SplitPdfProfile'
    convertServiceProps.append(convertImageProfile)
    convertImage['serviceProps'] = convertServiceProps

    # Now assign the files
    convertImageRequestItems = []
    convertImageRequestItem = {}

    convertImageRequestItem['nodeID'] = '1'
    convertImageRequestFiles = []

    convertImageFile = {}
    convertImageFile['value'] = FileID
    convertImageFile['contentType'] = ContentType
    
    convertImageRequestFiles.append(convertImageFile)
    convertImageRequestItem['files'] = convertImageRequestFiles

    convertImageRequestItems.append(convertImageRequestItem)
    convertImage['requestItems'] = convertImageRequestItems
    # Now do the posting
    # Now convert to Json
    convertImageJson = json.dumps(convertImage)
    #Now post
    convertImageRequest = requests.post(
    convertURL, data=convertImageJson, headers=serviceHeaders)
    result = json.loads(convertImageRequest.text)
    splitImages = result['resultItems'][0]['files']
    
    return splitImages

def process_image(splitImages,accessToken):
    serviceHeaders = {}
    serviceHeaders['Authorization'] = 'Bearer ' + accessToken
    serviceHeaders['Content-Type'] = 'application/hal+json; charset=utf-8'
    processImages = []
    processURL = baseURL + '/services/processimage'
    
    processImage = {}
    # First create the Service Props
    processServiceProps = []
    # We can also use this for the Classify & Extract request
    enviroment = {}
    enviroment['name'] = 'Env'
    enviroment['value'] = "D"
    processServiceProps.append(enviroment)

    processImageProfile = {}
    processImageProfile['name'] = 'Profile'
    processImageProfile['value'] = 'IntelligentFilling'
    processServiceProps.append(processImageProfile)
    processImageReturn = {}
    processImageReturn['name'] = 'ReturnFileDataInline'
    processImageReturn['value'] = 'false'
    processServiceProps.append(processImageReturn)
    processImage['serviceProps'] = processServiceProps

    # Now assign the files
    processImageRequestItems = []
    #Now try looping through all of the files
    node = 0
    for f in splitImages:

        processImageRequestItem = {}
        node += 1
        processImageRequestItem['nodeId'] = node
        processImageRequestFiles = []
        processValues = []
        
        fx = []
        fx.append(f)
        processImageRequestItem['files'] = fx
        processImageRequestItems.append(processImageRequestItem)
    processImage['requestItems'] = processImageRequestItems
    # Now do the posting
    # Now convert to Json
    processImageJson = json.dumps(processImage)
    #Now post
    processImageRequest = requests.post(
    processURL, data=processImageJson, headers=serviceHeaders)
    result = json.loads(processImageRequest.text)
    #Loop through all of the result items to put the files back at the correct level
    resultItems = result['resultItems']
    files = []
    for r in resultItems:
        files.append(r['files'][0])
    return files
def ocr_image(splitimages,accessToken):
    serviceHeaders = {}
    serviceHeaders['Authorization'] = 'Bearer ' + accessToken
    serviceHeaders['Content-Type'] = 'application/hal+json; charset=utf-8'
    ocrImages = []
    ocrURL = baseURL + '/services/fullpageocr'
    
    ocrImage = {}
    # First create the Service Props
    ocrServiceProps = []
    # We can also use this for the Classify & Extract request
    enviroment = {}
    enviroment['name'] = 'Env'
    enviroment['value'] = "D"
    ocrServiceProps.append(enviroment)

    ocrImageProfile = {}
    ocrImageProfile['name'] = 'OcrEngineName'
    ocrImageProfile['value'] = 'Advanced'
    ocrServiceProps.append(ocrImageProfile)
    ocrImage['serviceProps'] = ocrServiceProps

    # Now assign the files
    ocrImageRequestItems = []
    ocrImageRequestItem = {}

    ocrImageRequestItem['nodeId'] = 1
    ocrImageRequestFiles = []
    # Now specify the output type
    ocrOutputType = {}
    ocrOutputType['name'] = 'OutputType'
    ocrOutputType['value'] = 'text'
    ocrValues = []
    ocrValues.append(ocrOutputType)
    ocrImageRequestItem['values'] = ocrValues
    ocrImageRequestItem['files'] = splitimages
    ocrImageRequestItems.append(ocrImageRequestItem)
    ocrImage['requestItems'] = ocrImageRequestItems
    # Now do the posting
    # Now convert to Json
    ocrImageJson = json.dumps(ocrImage)
    #return ocrImageJson
    #Now post
    ocrImageRequest = requests.post(
    ocrURL, data=ocrImageJson, headers=serviceHeaders)
    result = json.loads(ocrImageRequest.text)
    return result
def get_text(textImage,accessToken):
    serviceHeaders = {}
    serviceHeaders['Authorization'] = 'Bearer ' + accessToken
    serviceHeaders['Content-Type'] = 'application/hal+json; charset=utf-8'

    #Now get the file ID from the textImage response
   # imageText = json.loads(textImage)
    src = textImage['resultItems'][0]['files'][0]['src']
    file = requests.get(src,headers=serviceHeaders)
    text = file.text
    return text
def remove_stop(text):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    words = word_tokenize(text)
    keepWords = []
    for r in words:
        if not r in stop_words:
            if r.isalpha():
                stemmed = porter.stem(r.lower())
                keepWords.append(stemmed)
    #convert it into a flat string
    kwstring = ' '.join(keepWords)
    return kwstring
def home(request):
    args = {'results':False}
    if request.method == 'POST' and "file_upload" in request.POST:
        try:
            myfile = request.FILES['myfile']
            #Save it temp
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            full_file_path = fs.location + "/" + filename
            #Guess the Mime Type
            mime = mimetypes.guess_type(full_file_path)
            escapes = ''.join([chr(char) for char in range(1, 32)])
            translator = str.maketrans('', '', escapes)
            #Now post the file to core capture services
            accessToken = auth()
            uploadFileID, uploadContentType = upload_image(full_file_path,accessToken).values()
            splitImages = image_convert(uploadFileID,uploadContentType,accessToken)
            processedImages = process_image(splitImages,accessToken)
            textImage = ocr_image(processedImages,accessToken)
            text = get_text(textImage,accessToken).translate('\r\n')
            document_text = text.translate(translator)
            #Clean the Text
            clean_text = remove_stop(document_text)
            #Send the Document Text to the TF-IDF
            #Load the TF-IDF
            loaded_vec = pickle.load(open("feature.sav", 'rb'))
            doc_vec = loaded_vec.transform([clean_text])
            #Now send the document Text to the svm
            #load the svm
            loaded_svm = pickle.load(open('svm.sav', 'rb'))
            predicted_doc = loaded_svm.predict(doc_vec)[0]
            #Now get the similar documents
            loaded_similar = pickle.load(open('similar.sav','rb'))
            X = pickle.load(open('TF-IDF.sav','rb'))
            loaded_similar['RecommendationScore'] = cosine_similarity(doc_vec, X).flatten()
            loaded_similar = loaded_similar.sort_values(by=['RecommendationScore'], ascending=False)
            loaded_similar['RecommendationScore'] = loaded_similar['RecommendationScore'].apply(lambda x: x*100)
            #Round up the score
            loaded_similar['RecommendationScore'] = loaded_similar['RecommendationScore'].apply(np.ceil)
            #Set it to and integer
            loaded_similar['RecommendationScore'] = loaded_similar['RecommendationScore'].astype(int)
            loaded_similar['RecommendationScore'] = loaded_similar['RecommendationScore'].apply(lambda x: 100 if x > 100 else x)
            index_name = loaded_similar.index.name
            column_names = loaded_similar.columns
            json_records = loaded_similar.to_json(orient='records')
            data = []
            data = json.loads(json_records)

            #Delete the file at the end
            fs.delete(filename)
            args = {'results':True,'predicted':predicted_doc,'table_data':data,'column_names':column_names,'index_name':index_name}
        except:
            #Delete the file at the end
            fs.delete(filename)
            args = {'error':True}
    return render(request, "if_gui/home.html",args)

def about(request):
    return render(request, "if_gui/about.html")

def contact(request):
    return render(request, "if_gui/contact.html")
    
    

def hello_there(request, name):
    return render(
        request,
        'if_gui/hello_there.html',
        {
            'name': name,
            'date': datetime.now()
        }
    )
print('http://127.0.0.1:8000/if_gui/VSCode')