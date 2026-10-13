"""
MongoDB Backend Connectivity Module
=====================================
Provides full CRUD operations for:
  - Student Records (students collection)
  - Prediction History / Audit Logs (predictions collection)
  - Model Metadata Snapshots (model_metadata collection)

Database: student_dropout_db
Connection: mongodb://localhost:27017

Usage:
    from database import db_manager
    db_manager.insert_prediction(student_data, result)
    history = db_manager.get_prediction_history(limit=50)
"""

import os
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import traceback


class MongoDBManager:
    """
    MongoDB Manager for Student Dropout Risk Analyzer.
    Handles all database operations including connection management,
    CRUD for students & predictions, and analytics queries.
    """

    def __init__(self, uri=None, db_name=None):
        """Initialize MongoDB connection with environment variable support."""
        self.uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.environ.get("MONGODB_DB_NAME", "student_dropout_db")
        self.client = None
        self.db = None
        self._connected = False
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB server."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Force a connection test
            self.client.server_info()
            self.db = self.client[self.db_name]
            self._connected = True
            self._ensure_indexes()
            print(f"[MongoDB] Connected successfully to '{self.db_name}' at {self.uri}")
        except Exception as e:
            self._connected = False
            print(f"[MongoDB] Connection failed: {e}")
            print("[MongoDB] App will run in offline mode (no database logging).")

    def _ensure_indexes(self):
        """Create indexes for optimal query performance."""
        if not self._connected:
            return
        try:
            # Predictions collection indexes
            self.db.predictions.create_index([("timestamp", DESCENDING)])
            self.db.predictions.create_index([("risk_level", 1)])
            self.db.predictions.create_index([("student_name", 1)])

            # Students collection indexes
            try:
                self.db.students.drop_index("student_id_1")
            except Exception:
                pass
            self.db.students.create_index([("student_id", 1)], unique=True, sparse=True)
            self.db.students.create_index([("department", 1)])

            # Model metadata
            self.db.model_metadata.create_index([("saved_at", DESCENDING)])

            print("[MongoDB] Indexes ensured on all collections.")
        except Exception as e:
            print(f"[MongoDB] Index creation warning: {e}")

    @property
    def is_connected(self):
        """Check if MongoDB connection is active."""
        return self._connected

    def get_connection_status(self):
        """Return detailed connection status for UI display."""
        if not self._connected:
            return {
                "status": "Disconnected",
                "color": "danger",
                "icon": "fas fa-times-circle",
                "detail": "MongoDB server not reachable"
            }
        try:
            info = self.client.server_info()
            return {
                "status": "Connected",
                "color": "success",
                "icon": "fas fa-check-circle",
                "detail": f"MongoDB v{info.get('version', 'N/A')} | DB: {self.db_name}",
                "server_version": info.get('version', 'N/A'),
                "database": self.db_name,
                "collections": self.db.list_collection_names()
            }
        except Exception:
            self._connected = False
            return {
                "status": "Disconnected",
                "color": "danger",
                "icon": "fas fa-times-circle",
                "detail": "Connection lost"
            }

    # ──────────────────────────────────────────────────────────
    # PREDICTION HISTORY (CREATE / READ / DELETE)
    # ──────────────────────────────────────────────────────────

    def insert_prediction(self, student_input, prediction_result, student_name=None):
        """
        Log a prediction to the predictions collection.

        Args:
            student_input: dict of raw student feature values
            prediction_result: dict returned by predict_student()
            student_name: optional student name for identification
        """
        if not self._connected:
            return None
        try:
            record = {
                "student_name": student_name or f"Student_{datetime.now().strftime('%H%M%S')}",
                "timestamp": datetime.now(),
                "input_features": student_input,
                "prediction": prediction_result.get("prediction", None),
                "prediction_label": prediction_result.get("prediction_label", ""),
                "probability": prediction_result.get("probability", 0.0),
                "probability_percent": prediction_result.get("probability_percent", 0.0),
                "risk_level": prediction_result.get("risk_level", ""),
                "risk_color": prediction_result.get("risk_color", ""),
                "threshold": prediction_result.get("threshold", 0.5),
                "risk_factors": prediction_result.get("risk_factors", []),
                "protective_factors": prediction_result.get("protective_factors", []),
                "domain_risks": prediction_result.get("domain_risks", {}),
                "interventions_count": len(prediction_result.get("interventions", [])),
                "student_summary": prediction_result.get("student_summary", {})
            }
            result = self.db.predictions.insert_one(record)
            print(f"[MongoDB] Prediction logged: {result.inserted_id} | Risk: {record['risk_level']}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"[MongoDB] Error inserting prediction: {e}")
            return None

    def get_prediction_history(self, limit=50):
        """Retrieve recent prediction history, newest first."""
        if not self._connected:
            return []
        try:
            cursor = self.db.predictions.find(
                {},
                {
                    "student_name": 1,
                    "timestamp": 1,
                    "prediction_label": 1,
                    "probability_percent": 1,
                    "risk_level": 1,
                    "risk_color": 1,
                    "input_features.Department": 1,
                    "input_features.CGPA": 1,
                    "input_features.Attendance_Rate": 1,
                    "student_summary": 1,
                    "interventions_count": 1
                }
            ).sort("timestamp", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[MongoDB] Error fetching history: {e}")
            return []

    def get_prediction_by_id(self, prediction_id):
        """Retrieve a single prediction record by its ObjectId."""
        if not self._connected:
            return None
        try:
            return self.db.predictions.find_one({"_id": ObjectId(prediction_id)})
        except Exception as e:
            print(f"[MongoDB] Error fetching prediction: {e}")
            return None

    def delete_prediction(self, prediction_id):
        """Delete a prediction record by its ObjectId."""
        if not self._connected:
            return False
        try:
            result = self.db.predictions.delete_one({"_id": ObjectId(prediction_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"[MongoDB] Error deleting prediction: {e}")
            return False

    def clear_prediction_history(self):
        """Clear all prediction history records."""
        if not self._connected:
            return 0
        try:
            result = self.db.predictions.delete_many({})
            print(f"[MongoDB] Cleared {result.deleted_count} prediction records.")
            return result.deleted_count
        except Exception as e:
            print(f"[MongoDB] Error clearing history: {e}")
            return 0

    # ──────────────────────────────────────────────────────────
    # STUDENT RECORDS (FULL CRUD)
    # ──────────────────────────────────────────────────────────

    def insert_student(self, student_data):
        """Insert a new student record."""
        if not self._connected:
            return None
        try:
            import random
            if "student_id" not in student_data or not student_data["student_id"]:
                student_data["student_id"] = f"STU_{random.randint(10000, 99999)}"
            student_data["created_at"] = datetime.now()
            student_data["updated_at"] = datetime.now()
            result = self.db.students.insert_one(student_data)
            print(f"[MongoDB] Student inserted: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"[MongoDB] Error inserting student: {e}")
            return None

    def get_all_students(self, limit=100):
        """Retrieve all student records."""
        if not self._connected:
            return []
        try:
            cursor = self.db.students.find().sort("created_at", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[MongoDB] Error fetching students: {e}")
            return []

    def update_student(self, student_id, update_data):
        """Update a student record."""
        if not self._connected:
            return False
        try:
            update_data["updated_at"] = datetime.now()
            result = self.db.students.update_one(
                {"_id": ObjectId(student_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"[MongoDB] Error updating student: {e}")
            return False

    def delete_student(self, student_id):
        """Delete a student record."""
        if not self._connected:
            return False
        try:
            result = self.db.students.delete_one({"_id": ObjectId(student_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"[MongoDB] Error deleting student: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # MODEL METADATA STORAGE
    # ──────────────────────────────────────────────────────────

    def save_model_metadata(self, metadata_dict):
        """Save a snapshot of model metadata to MongoDB."""
        if not self._connected:
            return None
        try:
            record = {
                "saved_at": datetime.now(),
                "best_model_name": metadata_dict.get("best_model_name", ""),
                "selected_threshold": metadata_dict.get("selected_threshold", 0.5),
                "total_dataset_size": metadata_dict.get("total_dataset_size", 0),
                "dropout_rate": metadata_dict.get("dropout_rate", 0.0),
                "imbalance_ratio": metadata_dict.get("imbalance_ratio", 0.0)
            }
            result = self.db.model_metadata.insert_one(record)
            print(f"[MongoDB] Model metadata saved: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"[MongoDB] Error saving metadata: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # ANALYTICS QUERIES
    # ──────────────────────────────────────────────────────────

    def get_prediction_stats(self):
        """Get aggregated statistics from prediction history."""
        if not self._connected:
            return {}
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_predictions": {"$sum": 1},
                        "avg_probability": {"$avg": "$probability_percent"},
                        "high_risk_count": {
                            "$sum": {"$cond": [{"$gte": ["$probability_percent", 60]}, 1, 0]}
                        },
                        "low_risk_count": {
                            "$sum": {"$cond": [{"$lt": ["$probability_percent", 30]}, 1, 0]}
                        }
                    }
                }
            ]
            result = list(self.db.predictions.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            return {"total_predictions": 0, "avg_probability": 0, "high_risk_count": 0, "low_risk_count": 0}
        except Exception as e:
            print(f"[MongoDB] Error computing stats: {e}")
            return {}

    def get_risk_distribution(self):
        """Get count of predictions by risk level."""
        if not self._connected:
            return []
        try:
            pipeline = [
                {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            return list(self.db.predictions.aggregate(pipeline))
        except Exception as e:
            print(f"[MongoDB] Error computing risk distribution: {e}")
            return []

    def get_collection_stats(self):
        """Get document counts for all collections."""
        if not self._connected:
            return {}
        try:
            return {
                "predictions": self.db.predictions.count_documents({}),
                "students": self.db.students.count_documents({}),
                "model_metadata": self.db.model_metadata.count_documents({})
            }
        except Exception as e:
            print(f"[MongoDB] Error getting collection stats: {e}")
            return {}

    def seed_sample_data(self, count=10):
        """Seed sample student predictions into MongoDB for demonstration."""
        if not self._connected:
            return 0
        import random
        from datetime import datetime, timedelta
        departments = ["CS", "ECE", "MECH", "CIVIL", "EEE", "IT"]
        genders = ["Male", "Female"]
        semesters = ["Year 1", "Year 2", "Year 3", "Year 4"]
        
        sample_names = [
            "Arjun Sharma", "Priya Patel", "Rahul Verma", "Sneha Iyer",
            "Karthik Nair", "Ananya Reddy", "Vikram Singh", "Divya Menon",
            "Rohan Joshi", "Meera Kulkarni", "Aditya Bose", "Pooja Das"
        ]
        
        inserted = 0
        now = datetime.now()
        for i in range(min(count, len(sample_names))):
            name = sample_names[i]
            dept = random.choice(departments)
            cgpa = round(random.uniform(1.2, 3.9), 2)
            attendance = round(random.uniform(50.0, 98.0), 1)
            study_hours = round(random.uniform(1.0, 7.0), 1)
            prob = round(random.uniform(10.0, 92.0), 1)
            
            if prob >= 70:
                risk_level = "High Risk"
                risk_color = "#f43f5e"
                pred_label = "High Dropout Risk"
                pred_code = 1
            elif prob >= 40:
                risk_level = "Moderate Risk"
                risk_color = "#f59e0b"
                pred_label = "Moderate Attrition Watch"
                pred_code = 0
            else:
                risk_level = "Low Risk"
                risk_color = "#10b981"
                pred_label = "Low Dropout Propensity"
                pred_code = 0

            time_offset = now - timedelta(minutes=random.randint(2, 180))
            
            doc = {
                "student_name": name,
                "timestamp": time_offset,
                "input_features": {
                    "Age": random.randint(18, 23),
                    "Gender": random.choice(genders),
                    "Family_Income": random.randint(18000, 75000),
                    "Internet_Access": "Yes" if random.random() > 0.2 else "No",
                    "Study_Hours_per_Day": study_hours,
                    "Attendance_Rate": attendance,
                    "Assignment_Delay_Days": random.randint(0, 6),
                    "Travel_Time_Minutes": random.randint(15, 60),
                    "Part_Time_Job": "Yes" if random.random() > 0.6 else "No",
                    "Scholarship": "Yes" if random.random() > 0.7 else "No",
                    "Stress_Index": round(random.uniform(2.0, 9.0), 1),
                    "GPA": round(cgpa - random.uniform(-0.3, 0.3), 2),
                    "Semester_GPA": round(cgpa - random.uniform(-0.4, 0.4), 2),
                    "CGPA": cgpa,
                    "Semester": random.choice(semesters),
                    "Department": dept,
                    "Parental_Education": random.choice(["Bachelor", "High School", "Master"])
                },
                "prediction": pred_code,
                "prediction_label": pred_label,
                "probability": prob / 100.0,
                "probability_percent": prob,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "threshold": 0.5,
                "risk_factors": ["Low Attendance" if attendance < 75 else "Academic Probation"] if prob >= 50 else [],
                "protective_factors": ["High CGPA" if cgpa > 3.0 else "Regular Attendance"] if prob < 50 else [],
                "domain_risks": {"Academic": "High" if cgpa < 2.5 else "Low", "Engagement": "Moderate"},
                "interventions_count": 2 if prob >= 50 else 0,
                "student_summary": {
                    "Academic_Average": cgpa,
                    "Academic_Trend": round(random.uniform(-0.5, 0.5), 2),
                    "Attendance_Rate": attendance
                }
            }
            self.db.predictions.insert_one(doc)
            inserted += 1
            
        print(f"[MongoDB] Seeded {inserted} demo predictions.")
        return inserted

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            print("[MongoDB] Connection closed.")


# ──────────────────────────────────────────────────────────
# SINGLETON INSTANCE — Import this in app.py
# ──────────────────────────────────────────────────────────
db_manager = MongoDBManager()
