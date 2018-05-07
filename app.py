from flask import Flask, request
import tensorflow as tf
from werkzeug import secure_filename
import json
import os
UPLOAD_FOLDER = 'uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
result = ""

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

with tf.gfile.FastGFile('model/output_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,name='')

sess = tf.Session()
label_lines = [line.rstrip() for line in tf.gfile.GFile('model/output_labels.txt')]
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

@app.route('/', methods=['GET'])
def display():
    global result
    if result == "":
        return "No Camera scan result to show!"
    else:
        return result


@app.route('/Scan', methods=['POST'])
def scan():
    global result
    global tf
    result = ""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('Received file : '+filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg'))
            image_data = tf.gfile.FastGFile('uploaded/upload.jpg','rb').read()

            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})*100
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            # for node_id in top_k:
            #     human_string = label_lines[node_id]
            #     score = predictions[0][node_id]
            #     result = result + '\n'+'%s (score = %.5f)' % (human_string, score)
            topScore1 = predictions[0][top_k[0]]
            result = "\nI am %2.2f % sure this is %s." % (topScore2,label_lines[top_k[0]])
            topScore2 = predictions[0][top_k[1]]
            if topScore1-topScore2 < 30 :
                result += "\nI am %2.2f % sure this is %s." % (topScore2,label_lines[top_k[1]])
    return result

if __name__ == '__main__':
    app.run()
