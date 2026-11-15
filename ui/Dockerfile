# Gaze maze UI. Two targets:
#
#   --target serve   production
#   --target dev     local development
#
#   # development (docker compose, or):
#   docker build -t gaze-ui-dev .
#   # production:
#   docker build --target serve -t gaze-ui .

# Build the UI (production dependency of the serve stage below)
FROM node:24-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Serve the UI (production)
FROM nginx:1.31-alpine AS serve

# nginx configuration
COPY <<'EOF' /etc/nginx/templates/default.conf.template
server {
    listen ${PORT};
    server_name _;
    root /usr/share/nginx/html;

    # Cloud Run logs itself, so nginx's access log not needed
    access_log off;

    gzip on;
    gzip_min_length 1024;
    gzip_types application/javascript text/css application/json application/wasm image/svg+xml;

    # Health check
    location = /health {
        default_type text/plain;
        return 200 "ok\n";
    }

    location = /config.json {
        default_type application/json;
        add_header Cache-Control "no-store" always;
        return 200 '{"modelUrl":"${MODEL_URL}"}';
    }

    # Content-hashed build output
    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }

    location /models/ {
        add_header Cache-Control "public, max-age=3600";
        try_files $uri =404;
    }

    location / {
        add_header Cache-Control "no-cache";
        try_files $uri $uri/ /index.html;
    }
}
EOF

COPY --from=build /app/dist /usr/share/nginx/html

ENV PORT=8080 \
    MODEL_URL=/models/gazenet.onnx

EXPOSE 8080

# Local development
FROM node:24-alpine AS dev
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
