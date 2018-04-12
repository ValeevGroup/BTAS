#ifndef __BTAS_BTAS_H
#define __BTAS_BTAS_H

#include <cassert>

// check prerequisite headers, in case btas is used as headers-only, bail if not
#define BTAS_SIGNAL_MISSING_PREREQUISITES
#include <btas/features.h>

#include <btas/generic/dot_impl.h>
#include <btas/generic/scal_impl.h>
#include <btas/generic/axpy_impl.h>
#include <btas/generic/ger_impl.h>
#include <btas/generic/gemv_impl.h>
#include <btas/generic/gemm_impl.h>
#include <btas/generic/gesvd_impl.h>
#include <btas/generic/cp_als.h>
#include <btas/generic/cp_rals.h>

#ifdef _CONTRACT_OPT_BAGEL
#include <btas/optimize/contract.h>
#else
#include <btas/generic/contract.h>
#endif

#endif // __BTAS_BTAS_H
