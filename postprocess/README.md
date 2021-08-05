# Experiments postprocessing data

Install `pyenv`

Use a python from pyenv to install `poetry`

```
pyenv install 3.8.11
pyenv shell 3.8.11
# follow poetry install instructions
```

configure poetry to use a local venv
```
poetry config virtualenvs.in-project true
```

set the local python version
```
pyenv local 3.8.11
```

This folder was initialized with
```
poetry init
```

I had an SSL error, which seems to be related to python not using the macos SSL certificates
eventually fixed it with

```
python
> print(ssl.get_default_verify_paths())
DefaultVerifyPaths(cafile='/usr/local/etc/openssl@1.1/cert.pem', capath='/usr/local/etc/openssl@1.1/certs', openssl_cafile_env='SSL_CERT_FILE', openssl_cafile='/usr/local/etc/openssl@1.1/cert.pem', openssl_capath_env='SSL_CERT_DIR', openssl_capath='/usr/local/etc/openssl@1.1/certs')
```

then `export REQUESTS_CA_BUNDLE=/usr/local/etc/openssl@1.1/cert.pem` before using poetry to install anything