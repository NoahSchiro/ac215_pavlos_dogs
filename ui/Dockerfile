# Stage 1: Build the React UI
FROM node:24-alpine AS builder

WORKDIR /app

# Install dependencies using the lockfile
COPY package.json package-lock.json ./
RUN npm ci

# Copy the application source and build it
COPY . .
RUN npm run build

# Stage 2: Serve the compiled UI
FROM nginx:1.31-alpine

# Rendered into /etc/nginx/conf.d/default.conf at startup from the variables
# below; see the template for what each one does.
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template

# Copy the production build into Nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Runtime configuration, overridden per deploy (deploy/deploy-ui.sh). One image
# serves any model: promoting a new one is a redeploy with a new MODEL_URL.
# PORT stays 80 here so `docker compose` is unchanged; Cloud Run sets 8080.
ENV PORT=80 \
    MODEL_URL=/models/gazenet.onnx \
    TELEMETRY=true \
    APP_VERSION=dev

# Expose the web server port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
