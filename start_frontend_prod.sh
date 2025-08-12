#!/bin/bash
# Script to build the React frontend for production and copy to backend static directory

# Navigate to the frontend directory
cd frontend

# Install dependencies (if not already installed)
npm install

# Build the frontend for production
npm run build

# Navigate back to the project root
cd ..

# Remove existing static directory in web_app (if any)
rm -rf web_app/static

# Create the static directory in web_app
mkdir -p web_app/static

# Copy the built frontend assets to web_app/static
cp -r frontend/dist/* web_app/static/
