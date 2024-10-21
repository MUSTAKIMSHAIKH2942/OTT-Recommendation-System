from flask import Flask, jsonify, request
from src.logging.logger import logger
from src.models.recommendation import recommend_movies

app = Flask(__name__)

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        recommendations = recommend_movies(user_id)
        return jsonify(recommendations.tolist()), 200
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
 
