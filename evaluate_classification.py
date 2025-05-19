"""
Comprehensive evaluation script for the article classification model.
This script evaluates the performance of our entity extraction and classification model
on a set of test articles with known categories.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.entity_extraction import extract_entities

# Sample Marathi news articles for testing with their expected categories
test_articles = [
    # Technology
    ("मुंबई: नवीन आयफोन 15 प्रो मॅक्स लाँच झाला आहे. या स्मार्टफोनमध्ये अॅपल ए17 प्रो चिप वापरली आहे. 5जी कनेक्टिव्हिटी आणि 48 मेगापिक्सेल कॅमेरा या फोनची वैशिष्ट्ये आहेत.", "Technology"),
    ("पुणे: गुगलने त्यांचे नवीन पिक्सेल फोन भारतात लाँच केले आहेत. या फोनमध्ये अँड्रॉइड 14 ऑपरेटिंग सिस्टम आहे. कृत्रिम बुद्धिमत्ता (एआय) वैशिष्ट्ये या फोनचे खास आकर्षण आहे.", "Technology"),
    
    # Entertainment
    ("पुणे: मराठी चित्रपट 'झुंज' ने बॉक्स ऑफिसवर धुमाकूळ घातला आहे. प्रसिद्ध अभिनेता रितेश देशमुख याने या चित्रपटात मुख्य भूमिका साकारली आहे. संगीत देखील उत्कृष्ट असल्याचे प्रेक्षकांचे मत आहे.", "Entertainment"),
    ("मुंबई: बॉलिवूड अभिनेता शाहरुख खान यांचा नवीन चित्रपट 'पठाण' प्रदर्शित झाला आहे. या चित्रपटाला प्रेक्षकांचा उत्तम प्रतिसाद मिळत आहे. दीपिका पदुकोण आणि जॉन अब्राहम यांनीही या चित्रपटात महत्त्वाच्या भूमिका साकारल्या आहेत.", "Entertainment"),
    
    # Sports
    ("नागपूर: भारतीय क्रिकेट संघाने ऑस्ट्रेलियाविरुद्ध पहिल्या कसोटी सामन्यात दमदार विजय मिळवला. विराट कोहलीने शतक झळकावले, तर जसप्रीत बुमराहने 5 विकेट घेतल्या.", "Sports"),
    ("मुंबई: आयपीएल 2023 मध्ये मुंबई इंडियन्सने चेन्नई सुपर किंग्जला हरवले. रोहित शर्माने कर्णधार म्हणून उत्कृष्ट नेतृत्व केले. सूर्यकुमार यादवने नाबाद 80 धावा केल्या.", "Sports"),
    
    # Politics
    ("दिल्ली: लोकसभा निवडणुकीसाठी आचारसंहिता लागू झाली आहे. सर्व राजकीय पक्षांनी प्रचाराला सुरुवात केली असून, पंतप्रधान नरेंद्र मोदी यांनी आज मोठी सभा घेतली.", "Politics"),
    ("मुंबई: महाराष्ट्रात विधानसभा निवडणुकीची तयारी सुरू झाली आहे. विरोधी पक्षांनी सरकारवर टीका केली असून, सत्ताधारी पक्षाने त्यांच्या विकासकामांचा प्रचार सुरू केला आहे.", "Politics"),
    
    # Business
    ("मुंबई: शेअर बाजारात आज मोठी तेजी दिसून आली. सेन्सेक्समध्ये 500 अंकांची वाढ झाली. बँकिंग आणि आयटी क्षेत्रातील कंपन्यांच्या शेअर्समध्ये मोठी खरेदी झाल्याचे दिसून आले.", "Business"),
    ("नवी दिल्ली: रिझर्व्ह बँकेने व्याजदरात 0.25 टक्क्यांची वाढ केली आहे. या निर्णयामुळे गृहकर्ज आणि वाहन कर्ज महागणार आहे. बँकांनी त्यांच्या व्याजदरात वाढ करण्याची शक्यता आहे.", "Business"),
    
    # Health
    ("पुणे: कोरोना विषाणूचा नवीन व्हेरिएंट आढळला आहे. डॉक्टरांनी नागरिकांना मास्क वापरण्याचा सल्ला दिला आहे. रुग्णालयांमध्ये विशेष कक्ष सज्ज करण्यात आले आहेत.", "Health"),
    ("मुंबई: डेंग्यूचे रुग्ण वाढत आहेत. डॉक्टरांनी नागरिकांना सावधगिरी बाळगण्याचा सल्ला दिला आहे. पाणी साठवून न ठेवण्याचे आवाहन करण्यात आले आहे.", "Health"),
    
    # Education
    ("औरंगाबाद: राज्य शिक्षण मंडळाने दहावी आणि बारावीच्या परीक्षांचे वेळापत्रक जाहीर केले आहे. विद्यार्थ्यांना अभ्यासासाठी पुरेसा वेळ मिळावा यासाठी परीक्षा मार्च महिन्यात घेण्यात येणार आहेत.", "Education"),
    ("पुणे: शैक्षणिक वर्ष 2023-24 साठी नवीन अभ्यासक्रम जाहीर करण्यात आला आहे. डिजिटल शिक्षणावर भर देण्यात आला असून, विद्यार्थ्यांना प्रात्यक्षिक ज्ञान देण्यावर भर दिला जाणार आहे.", "Education"),
    
    # Environment
    ("नाशिक: वृक्षतोडीमुळे पर्यावरणावर विपरीत परिणाम होत आहे. जागतिक तापमानवाढ रोखण्यासाठी जनजागृती मोहीम राबवण्यात येत आहे. वनविभागाने वृक्षारोपण कार्यक्रम हाती घेतला आहे.", "Environment"),
    ("मुंबई: समुद्रातील प्लास्टिक प्रदूषणाबाबत चिंता व्यक्त करण्यात आली आहे. समुद्री जीवांवर याचा विपरीत परिणाम होत असल्याचे शास्त्रज्ञांनी सांगितले आहे. प्लास्टिक वापर कमी करण्याचे आवाहन करण्यात आले आहे.", "Environment"),
    
    # Crime
    ("ठाणे: शहरात दरोड्याची घटना घडली. चोरट्यांनी एका ज्वेलरी दुकानात प्रवेश करून सुमारे 50 लाखांचे दागिने लंपास केले. पोलिसांनी तपास सुरू केला असून, सीसीटीव्ही फुटेज तपासले जात आहे.", "Crime"),
    ("पुणे: सायबर गुन्हेगारीत वाढ होत आहे. अनेक नागरिकांना ऑनलाइन फसवणूक करून लाखो रुपयांना गंडा घालण्यात आला आहे. पोलिसांनी नागरिकांना सतर्क राहण्याचा सल्ला दिला आहे.", "Crime"),
    
    # Agriculture
    ("नागपूर: अवकाळी पावसामुळे शेतकऱ्यांचे मोठे नुकसान झाले आहे. विदर्भातील कापूस पिकाचे नुकसान झाले असून, शेतकऱ्यांना नुकसान भरपाई देण्याची मागणी होत आहे. कृषी विभागाने पंचनामे सुरू केले आहेत.", "Agriculture"),
    ("औरंगाबाद: कृषी विभागाने नवीन बियाणे विकसित केले आहे. या बियाण्यांमुळे कमी पाण्यात अधिक उत्पादन घेता येणार आहे. शेतकऱ्यांना या बियाण्यांचे वाटप करण्यात येणार आहे.", "Agriculture")
]

# Mixed content articles
mixed_articles = [
    ("मुंबई: राज्यात कोरोनाचा प्रादुर्भाव वाढत असताना शेअर बाजारात मोठी घसरण झाली आहे. शिक्षण विभागाने शाळा बंद ठेवण्याचा निर्णय घेतला आहे. क्रिकेट सामने रद्द करण्यात आले आहेत.", "Mixed"),
    ("पुणे: शहरात क्रिकेट स्पर्धेदरम्यान दंगल झाली. पोलिसांनी परिस्थिती नियंत्रणात आणली. स्पर्धा पुढे ढकलण्यात आली असून, क्रीडा विभागाने चौकशीचे आदेश दिले आहेत.", "Mixed"),
    ("नागपूर: शेतकऱ्यांनी केलेल्या आंदोलनामुळे राजकीय वातावरण तापले आहे. विरोधी पक्षांनी सरकारवर टीका केली असून, शेतकऱ्यांच्या मागण्या मान्य करण्याची मागणी केली आहे.", "Mixed")
]

def evaluate_classification():
    """
    Evaluate the performance of the classification model on the test articles.
    """
    # Prepare data
    texts = [article for article, _ in test_articles]
    true_categories = [category for _, category in test_articles]
    
    # Predict categories
    predicted_categories = []
    for text in texts:
        entities = extract_entities(text)
        predicted_categories.append(entities['category'])
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_categories, predicted_categories) if true == pred)
    accuracy = (correct / len(true_categories)) * 100
    
    # Generate classification report
    unique_categories = sorted(list(set(true_categories + predicted_categories)))
    report = classification_report(true_categories, predicted_categories, labels=unique_categories, zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_categories, predicted_categories, labels=unique_categories)
    
    # Print results
    print("=" * 80)
    print("CLASSIFICATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {correct}/{len(true_categories)} ({accuracy:.2f}%)")
    print("\nClassification Report:")
    print(report)
    
    # Print detailed results for each article
    print("\nDetailed Results:")
    print("-" * 80)
    for i, ((text, true_category), pred_category) in enumerate(zip(test_articles, predicted_categories)):
        is_correct = true_category == pred_category
        result_mark = "✓" if is_correct else "✗"
        print(f"Article {i+1}: {text[:50]}...")
        print(f"  - Expected: {true_category}")
        print(f"  - Predicted: {pred_category} {result_mark}")
        print("-" * 80)
    
    # Test with mixed content
    print("\nMixed Content Articles:")
    print("=" * 80)
    for i, (text, _) in enumerate(mixed_articles):
        entities = extract_entities(text)
        print(f"Mixed Article {i+1}: {text[:50]}...")
        print(f"  - Detected Category: {entities['category']}")
        print(f"  - Emotions: {entities['emotions']['dominant']}")
        print(f"  - Severity: {entities['severity']}")
        print("-" * 80)
    
    # Try to plot confusion matrix if running in an environment with display support
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_categories, yticklabels=unique_categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"\nCould not generate confusion matrix plot: {e}")

if __name__ == "__main__":
    evaluate_classification()
