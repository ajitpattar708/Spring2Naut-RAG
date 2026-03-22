plugins {
    java
    id("io.micronaut.application") version "4.10.8"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.micronaut:micronaut-http-server-netty:4.10.8")
    implementation("io.micronaut:micronaut-runtime:4.10.8")
}
