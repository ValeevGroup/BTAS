set(BTAS_TRACKED_VGCMAKEKIT_TAG 256d9462bb765787f5acb69be154b26d6efba8b6)

# oldest Boost we can tolerate ... likely can use an earlier version, but:
# - as of oct 2023 tested with 1.71 and up only
# - avoids the need to avoid 1.70 in which Boost.Container is broken
# - matches the version provided by https://github.com/Orphis/boost-cmake as of oct 2023
set(BTAS_OLDEST_BOOST_VERSION 1.71)
