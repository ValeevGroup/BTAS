set(BTAS_TRACKED_VGCMAKEKIT_TAG bbcf1ec7eaabe9b9578e95bb2c28c9a120a55e1a)

# oldest Boost we can tolerate ... likely can use an earlier version, but:
# - as of oct 2023 tested with 1.71 and up only
# - avoids the need to avoid 1.70 in which Boost.Container is broken
# - matches the version provided by https://github.com/Orphis/boost-cmake as of oct 2023
set(BTAS_OLDEST_BOOST_VERSION 1.71)
