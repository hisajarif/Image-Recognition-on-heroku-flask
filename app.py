from flask import Flask, render_template, request
import tensorflow as tf
import os
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with tf.gfile.FastGFile('model/output_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,name='')

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)

    return 'success!'

@app.route('/predict')
def predict():
    result = ""
    global tf
    image_data = tf.gfile.FastGFile('model/Arduino.jpg','rb').read()
    label_lines = [line.rstrip() for line in tf.gfile.GFile('model/output_labels.txt')]
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result = result + '\n'+'%s (score = %.5f)' % (human_string, score)# human_string +' score = '+str(score)
            #print('%s (score = %.5f)' % (human_string, score))
    return result

if __name__ == '__main__':
    app.run()
