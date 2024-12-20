"""
Parameter for cloud job submission and DocumentDB access. All parameters are required.
"""

# Required by Batch job submission
JOB_QUEUE = ""
JOB_DEFINITION_PICKING = ""
JOB_DEFINITION_ASSOCIATION = ""

# Required by DocumentDB access
# please verify username, password, and location of the pem file
DOCDB_ENDPOINT_URI = ""

# Required by EarthScope S3 access
EARTHSCOPE_S3_ACCESS_POINT = ""
