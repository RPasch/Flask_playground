#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request , redirect, url_for
import matplotlib.pyplot as plt
from io import StringIO , BytesIO
import pandas as pd
from sklearn import manifold
import numpy as np
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad
from time import time
import networkx as nx
import base64
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import pairwise_distances
import scipy as sc

import seaborn as sns
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)

class DataStore():
    l = None
    m = None

data = DataStore()

app_data = {
    "name": "Predicting Football Rankings Web App",
    "description": "A football prediction app",
    "author": "Joannes Pasch",
    "html_title": "Predicting Rankings",
    "project_name": "Predicting EPL rankings using deep similarity learning",
    "keywords": "predicting, similarity, app, football"
}


class CCA():
    def __init__(self, components, lmbda, max_iter=100):
        self.components = components
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.cca_loss_grad = grad(self.cca_loss)

    def fit_transform(self, X, y=None):
        self.n = X.shape[0]
        Y = PCA(self.components).fit_transform(X)  # np.random.randn(self.n, self.k)
        P = pairwise_distances(X)
        self.P_tril = P[np.tril_indices(P.shape[0], k=-1)]
        print("Initial loss:", self.cca_loss(Y))
        res = minimize(
            self.cca_loss, Y.reshape(self.n * self.components), method='L-BFGS-B',
            jac=self.cca_loss_grad, options={'disp': True, 'gtol': 1e-9, 'maxiter': self.max_iter})
        Y = res.x.reshape(self.n, self.components)
        print("Trained loss:", self.cca_loss(Y))
        return Y

    def cca_loss(self, Y):
        Y = Y.reshape(self.n, self.components)
        # use dist squared since derivative of sqrt at zero is inf
        D = (np.sum((Y[None, :] - Y[:, None]) ** 2, -1))
        # now we take square root of the off diagnals
        D_tril = np.sqrt(D[np.tril_indices(D.shape[0], k=-1)])
        return (((self.P_tril - D_tril) ** 2) * (D_tril < self.lmbda)).sum()


class CDA():
    def __init__(self, components, k, lmbda, max_iter=100):
        self.components = components
        self.k = k
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.cda_loss_grad = grad(self.cda_loss)

    def fit_transform(self, X, y=None):
        self.n = X.shape[0]
        tree = sc.spatial.KDTree(X)
        dist, idx = tree.query(X, self.k + 1)
        G = nx.Graph()
        G.add_nodes_from(idx[:, 0])
        for i in range(1, self.k + 1):
            G.add_weighted_edges_from(np.c_[idx[:, [0, i]], dist[:, i]])
        P = np.asarray(nx.floyd_warshall_numpy(G))
        Y = KernelPCA(self.components, kernel='precomputed').fit_transform(-0.5 * (P ** 2.0))
        self.P_tril = P[np.tril_indices(P.shape[0], k=-1)]
        print("Initial loss:", self.cda_loss(Y))
        res = minimize(
            self.cda_loss, Y.reshape(self.n * self.components), method='L-BFGS-B',
            jac=self.cda_loss_grad, options={'disp': True, 'gtol': 1e-9, 'maxiter': self.max_iter})
        Y = res.x.reshape(self.n, self.components)
        print("Trained loss:", self.cda_loss(Y))
        return Y

    def cda_loss(self, Y):
        Y = Y.reshape(self.n, self.components)
        # use dist squared since derivative of sqrt at zero is inf
        D = (np.sum((Y[None, :] - Y[:, None]) ** 2, -1))
        # now we take square root of the off diagnals
        D_tril = np.sqrt(D[np.tril_indices(D.shape[0], k=-1)])
        return (((self.P_tril - D_tril) ** 2) * (D_tril < self.lmbda)).sum()


@app.route('/')
def index():
    return render_template('index.html', app_data=app_data)


@app.route('/predict', methods=['GET', 'POST'])
def b1():
    if request.method == 'POST':
        print(request.form['League'])
        data.l = request.form['League']
        data.m = request.form['Method']
        print(request.form['Method'])
        return redirect(url_for('prediction',  l = data.l, m = data.m))
        # show_pred(request.form['League'], request.form['Method'])
    return render_template('predict.html', app_data=app_data)


@app.route('/stats')
def b2():
    return render_template('stats.html', app_data=app_data)


@app.route('/info')
def b3():
    return render_template('info.html', app_data=app_data)

@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
    img = BytesIO()
    y = [1, 2, 3, 4, 5]
    x = [0, 2, 1, 3, 4]
    plt.figure()
    plt.plot(x, y)
    plt.savefig('templates/output1.png',format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())



    df = pd.DataFrame({'Patient Name': ["Some name", "Another name"],
                       "Patient ID": [123, 456],
                       "Misc Data Point": [8, 53]})


    df = pd.read_pickle('sim_df_epl.pkl')
    if data.m == 'MDS':
        df_rank = getMDS(df)
    if data.m == 'CCA':
        df_rank = getCCA(df)
    if data.m == 'CDA':
        df_rank = getCDA(df)
    if data.m == 'Isomap':
        df_rank = getIso(df)
    return render_template('prediction.html',plot_url = plot_url ,app_data=app_data, m = data.m, l = data.l,
                           column_names=df_rank.columns.values,row_data=list(df_rank.values.tolist()),link_column="True",
                           zip=zip)

def getMDS(df):
    names = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves',
             'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle',
             'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
    sim_arr = np.array(np.array(df))
    sim_arr -= sim_arr.mean()
    mds_1d = manifold.MDS(n_components=1, max_iter=3000, eps=1e-2, random_state=1, dissimilarity="precomputed",n_jobs=1)
    sim_fit_1d = mds_1d.fit_transform(sim_arr)
    teams_pts = []
    teams_pts_arr = np.zeros((2,20))
    for i in range(len(sim_fit_1d)):
        # teams_pts_arr[i,0] = sim_fit_1d[i]
        # teams_pts_arr[i,1] = names[i]
        teams_pts.append((sim_fit_1d[i], names[i]))
    teams_pts.sort(reverse = True)
    list_names = []
    for i in range(len(sim_fit_1d)):
        list_names.append(teams_pts[i][1])
    df_f = pd.DataFrame(columns= ['True','Predicted'])
    df_f['True'] = names
    print(teams_pts)
    df_f['Predicted'] = list_names
    return df_f

def getCCA(df):
    names = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves',
             'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle',
             'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
    sim_arr = np.array(np.array(df))
    sim_arr -= sim_arr.mean()
    cca = CCA(1, 0.1, max_iter=10)
    cca_fit = cca.fit_transform(sim_arr)
    teams_pts = []
    teams_pts_arr = np.zeros((2,20))
    for i in range(len(cca_fit)):
        # teams_pts_arr[i,0] = sim_fit_1d[i]
        # teams_pts_arr[i,1] = names[i]
        teams_pts.append((cca_fit[i], names[i]))
    teams_pts.sort(reverse = True)
    list_names = []
    for i in range(len(cca_fit)):
        list_names.append(teams_pts[i][1])
    df_f = pd.DataFrame(columns= ['True','Predicted'])
    df_f['True'] = names
    print(teams_pts)
    df_f['Predicted'] = list_names
    return df_f


def getCDA(df):
    names = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves',
             'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle',
             'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
    sim_arr = np.array(np.array(df))
    sim_arr -= sim_arr.mean()
    cda = CDA(2, 5, 0.0005, max_iter=10)
    cda_fit = cda.fit_transform(sim_arr)

    cda_fit_1 = CDA(1, 5, 0.0005, max_iter=10).fit_transform(cda_fit)

    teams_pts = []
    teams_pts_arr = np.zeros((2, 20))
    for i in range(len(cda_fit_1)):
        # teams_pts_arr[i,0] = sim_fit_1d[i]
        # teams_pts_arr[i,1] = names[i]
        teams_pts.append((cda_fit_1[i], names[i]))
    teams_pts.sort(reverse=True)
    list_names = []
    for i in range(len(cda_fit_1)):
        list_names.append(teams_pts[i][1])
    df_f = pd.DataFrame(columns=['True', 'Predicted'])
    df_f['True'] = names
    print(teams_pts)
    df_f['Predicted'] = list_names
    return df_f

def getIso(df):
    names = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves',
             'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle',
             'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
    sim_arr = np.array(np.array(df))
    sim_arr -= sim_arr.mean()
    iso = manifold.Isomap(5, 1)
    iso_fit = iso.fit_transform(sim_arr)
    teams_pts = []
    teams_pts_arr = np.zeros((2,20))
    for i in range(len(iso_fit)):
        # teams_pts_arr[i,0] = sim_fit_1d[i]
        # teams_pts_arr[i,1] = names[i]
        teams_pts.append((iso_fit[i], names[i]))
    teams_pts.sort(reverse = True)
    list_names = []
    for i in range(len(iso_fit)):
        list_names.append(teams_pts[i][1])
    df_f = pd.DataFrame(columns= ['True','Predicted'])
    df_f['True'] = names
    print(teams_pts)
    df_f['Predicted'] = list_names
    return df_f

# ------- PRODUCTION CONFIG -------
# if __name__ == '__main__':
#    app.run()

def show_pred(method, league):
    print(method, league)
    return render_template('info.html', app_data=app_data)


# ------- DEVELOPMENT CONFIG -------
if __name__ == '__main__':
    app.run(debug=True)
