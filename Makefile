.PHONY: test package clean unittest

PYTHON ?= $(shell which python)
PYTEST ?= $(shell which pytest)

PROJ_DIR := $(shell readlink -f ${CURDIR})
DIST_DIR := ${PROJ_DIR}/dist
TEST_DIR := ${PROJ_DIR}/test
SRC_DIR  := ${PROJ_DIR}/bayes_opt

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
clean:
	rm -rf ${DIST_DIR}

test: unittest

unittest:
	$(PYTEST) "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)
