# import os
# from flask import Flask
# from flask import render_template
# from flask import request
# from flask import send_from_directory
#
# from ldamodel import *
#
#
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def index():
#     return render_template(
#         'home.html',
#         data=[
#             {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/new_data'},
#             {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/data/20news-bydate-train/comp.graphics'},
#             {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/data/20news-bydate-train/comp.os.ms-windows.misc'},
#             {'name':'comp.sys.ibm.pc.hardware'}
#         ])
#
# @app.route("/upload" , methods=['GET', 'POST'])
# def upload():
#
#    try:
#         TRAIN_DATA = request.form.get('comp_select')
#         TRAIN_OUT_FILE =" ../newdata1/merged_file.json"
#         topic = request.form.get('topics')
#
#         merge_tojsonl(TRAIN_DATA, TRAIN_OUT_FILE)
#         print topic
#
#         return(str(topic)) # just to see what select is
#    except:
#        print "errror"
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#     # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
