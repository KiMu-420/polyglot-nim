package main

import (
    "log"
    "net/http"
    "net/http/httputil"
    "net/url"
    "os"
    "path/filepath"
)

func main() {
    target, _ := url.Parse("http://localhost:8080")
    proxy := httputil.NewSingleHostReverseProxy(target)

    execPath, _ := os.Getwd()
    projectRoot := filepath.Dir(execPath)
    staticPath := filepath.Join(projectRoot, "static")

    http.Handle("/api/", proxy)
    http.Handle("/", http.FileServer(http.Dir(staticPath)))

    log.Printf("Serving static from: %s", staticPath)
    log.Println("Go gateway listening on :8081")
    log.Fatal(http.ListenAndServe(":8081", nil))
}
