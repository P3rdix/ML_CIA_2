from flask import Flask, request, render_template
import pymysql as pms
import pandas as pd
import pickle

app = Flask(__name__)



@app.route("/")
def main():
    return render_template("index.html", arg = "")



@app.route("/login", methods=['Post'])
def attempt_login():
    con = pms.connect(host="localhost",
                  port = 3306,
                  user = 'mlcia',
                  password='abcd',
                  db='ml_cia')
    select = "select * from accounts"
    data = pd.read_sql(select,con)
    usr = [i for i in (request.form.values())]
    d = data[data["user_name"] == usr[0]]
    l = d["password"].values
    if l == usr[1]:
        return render_template("model.html")
    else:
        return render_template("index.html",arg = "Invalid username or password")



@app.route("/registry", methods=['Get','Post'])
def registry(s = ""):
    return render_template("registry.html", arg2 = s)



@app.route("/register_user", methods = ['Post'])
def register_user():
    con = pms.connect(host="localhost",
                  port = 3306,
                  user = 'mlcia',
                  password='abcd',
                  db='ml_cia')
    select = "select * from accounts"
    data = pd.read_sql(select,con)
    new_usr = [i for i in (request.form.values())]
    if new_usr[0] in data["user_name"] or new_usr[1] in data["email"]:
        return registry(s = "Username or Email are already used")
    else:
        if len(new_usr[2]) > 5:
            insert = "insert into accounts values('{}','{}','{}')".format(new_usr[0],new_usr[1],new_usr[2])
            cur = con.cursor()
            cur.execute(insert)
            return render_template("index.html", arg = "")
        else:
            return registry(s = "Password must have more than 5 characters")


@app.route("/result", methods = ["Post"])
def result():
    model = pickle.load(open('model.pkl','rb'))
    inp = [i for i in (request.form.values())]
    out = model.predict(inp[0],inp[1])
    return render_template("result.html", sentence = out)


if __name__ == '__main__':
    app.run(host = 'localhost')