from flask import Flask, render_template

app = Flask(__name__)

# 路由 1：系统大屏首页 (登录后看到的)
@app.route('/')
def index():
    return render_template('index.html')

# 路由 2：登录/注册页 (独立页面，不带侧边栏)
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/data')
def data_manage():
    return render_template('data_manage.html')

# ================================
# 相似检索页路由
# ================================
@app.route('/search')
def similarity_search():
    return render_template('search.html')

# ================================
# 用户管理页路由 (仅管理员可见)
# ================================
@app.route('/users')
def user_management():
    return render_template('users.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

# ================================
# 性能评测页路由
# ================================
@app.route('/benchmark')
def benchmark():
    return render_template('benchmark.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)